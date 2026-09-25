#!/usr/bin/env python3
"""One Ray node per Fuyao PyTorchJob pod; only node zero runs the trainer.

Bootstrap follows Ray's existing-cluster protocol:
https://docs.ray.io/en/latest/ray-core/starting-ray.html
All pods run this same command with Fuyao's NODE_RANK/RANK and MASTER_ADDR.
No SSH, nested torchrun, independent per-node trainer, or CUDA remapping.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid


TRANSIENT_FILE_ERRORS = {errno.ESTALE, errno.ENOENT, errno.EIO, errno.ETIMEDOUT, errno.EINTR, errno.EAGAIN}


def _status_io(operation, description, attempts=4):
    """Reopen on each attempt: atomic rename on shared storage can stale an fd.

    A short storage interruption is not evidence that the trainer died. Return
    unavailable after bounded retries; callers enforce a separate grace period.
    Permission/configuration errors are deliberately not swallowed.
    """
    for attempt in range(attempts):
        try:
            return True, operation()
        except OSError as exc:
            if exc.errno not in TRANSIENT_FILE_ERRORS:
                raise
            error = exc
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(.1 * 2 ** attempt)
    print(f'[multi-node] transient {description} unavailable after {attempts} attempts: {error}', flush=True)
    return False, None


def topology(env):
    nodes = int(env['PA_OPD_NNODES'])
    local_gpus = int(env['PA_OPD_GPUS_PER_NODE'])
    rank = int(env.get('NODE_RANK', env.get('RANK', '-1')))
    master = env.get('MASTER_ADDR', '')
    port = int(env.get('PA_OPD_RAY_PORT', env.get('MASTER_PORT', '6379')))
    if nodes < 2 or local_gpus < 1 or not 0 <= rank < nodes:
        raise ValueError(f'Invalid multi-node topology: nodes={nodes}, gpus={local_gpus}, rank={rank}')
    if not master or master in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Multi-node Ray requires the routable Fuyao MASTER_ADDR, not localhost')
    if not 0 < port < 65536:
        raise ValueError('Invalid Ray head port')
    return nodes, local_gpus, rank, master, port


def write_status(path, state, **extra):
    payload = json.dumps(dict(state=state, heartbeat=time.time(), sequence=time.monotonic_ns(), **extra))

    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Each retry uses a fresh inode/name; a prior successful rename may
        # have been followed by a transient client-side error.
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            temporary.write_text(payload)
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Only a failed-write temporary may remain; never delete status.json.

    available, _ = _status_io(write, 'status write')
    return available


def read_status(path):
    def read():
        status = json.loads(path.read_text())
        if not isinstance(status, dict) or status.get('state') not in ('starting', 'training', 'finished'):
            raise ValueError('Invalid supervisor status')
        if not isinstance(status.get('sequence'), int) or status['sequence'] < 0:
            raise ValueError('Missing supervisor heartbeat sequence')
        if status['state'] == 'finished' and not isinstance(status.get('returncode'), int):
            raise ValueError('Missing supervisor exit code')
        return status

    available, status = _status_io(read, 'status read')
    return status if available else {}


class HeartbeatWatchdog:
    """Use elapsed local monotonic time, never compare clocks on two nodes."""

    def __init__(self, timeout=300):
        self.timeout = timeout
        self.last_progress = time.monotonic()
        self.sequence = -1

    def observe(self, status):
        sequence = status.get('sequence', -1)
        if sequence > self.sequence:
            self.sequence = sequence
            self.last_progress = time.monotonic()
        if time.monotonic() - self.last_progress > self.timeout:
            raise TimeoutError('No advancing Ray head heartbeat within the storage grace period')


class HeartbeatWriter:
    def __init__(self, path, timeout=300):
        self.path = path
        self.timeout = timeout
        self.last_success = time.monotonic()

    def publish(self, state, **extra):
        if write_status(self.path, state, **extra):
            self.last_success = time.monotonic()
            return True
        if time.monotonic() - self.last_success > self.timeout:
            raise TimeoutError('Cannot publish Ray head heartbeat within the storage grace period')
        return False


def verify_final_checkpoint(output, world_size):
    from scripts.validate_checkpoint import validate

    step = int((output / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip())
    checkpoint = output / 'checkpoints' / f'global_step_{step}'
    errors = validate(checkpoint, world_size)
    export = output / 'inference' / f'global_step_{step}'
    index = export / 'model.safetensors.index.json'
    weights = (set(json.loads(index.read_text())['weight_map'].values()) if index.is_file()
               else {'model.safetensors'})
    for name in weights | {'config.json', 'tokenizer.json', 'preprocessor_config.json'}:
        path = export / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing/empty inference file: {path}')
    if errors:
        raise RuntimeError('Final checkpoint validation failed:\n' + '\n'.join(errors))
    print(f'[multi-node] final inference and {world_size}-rank resume checkpoint verified: step={step}', flush=True)


def main():
    command = sys.argv[1:]
    if command and command[0] == '--':
        command.pop(0)
    if not command:
        raise ValueError('A trainer command is required after --')
    nodes, local_gpus, rank, master, port = topology(os.environ)
    launch_id = os.environ['PA_OPD_LAUNCH_ID']
    if not launch_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in launch_id):
        raise ValueError('Unsafe PA_OPD_LAUNCH_ID')
    status_path = Path(os.environ['OUTPUT_DIR']) / 'launcher' / launch_id / 'status.json'
    address = f'{master}:{port}'
    # Ray/FSDP establishes its own per-actor ranks. Do not leak pod ranks into
    # model imports or vLLM engine child processes.
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(name, None)
    os.environ['RAY_ADDRESS'] = address
    os.environ.setdefault('RAY_USAGE_STATS_ENABLED', '0')
    deadline = time.monotonic() + int(os.environ.get('PA_OPD_CLUSTER_TIMEOUT', '900'))
    child = None

    def interrupted(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = 1
    owns_status = False
    writer = HeartbeatWriter(status_path)
    try:
        if rank == 0:
            if status_path.exists():
                raise RuntimeError('Launch id already exists; submit with a fresh PA_OPD_LAUNCH_ID')
            writer.publish('starting')
            owns_status = True
        else:
            while True:
                status = read_status(status_path)
                if status.get('state') == 'finished':
                    return int(status['returncode'])
                try:
                    with socket.create_connection((master, port), timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Timed out waiting for the Ray head')
                    time.sleep(2)

        # Every pod gets exactly one raylet advertising its own 16 devices.
        local_ip = socket.gethostbyname(socket.gethostname())
        ray_command = [sys.executable, '-m', 'ray.scripts.scripts', 'start',
                       f'--node-ip-address={local_ip}', f'--num-gpus={local_gpus}',
                       '--disable-usage-stats']
        if rank == 0:
            ray_command += ['--head', f'--port={port}', '--include-dashboard=false',
                            f'--temp-dir={os.environ["RAY_TMPDIR"]}']
        else:
            ray_command += [f'--address={address}']
        print(f'[multi-node] node={rank}/{nodes}, ip={local_ip}, head={address}', flush=True)
        subprocess.run(ray_command, check=True, timeout=180)

        if rank != 0:
            watchdog = HeartbeatWatchdog()
            while True:
                status = read_status(status_path)
                if status.get('state') == 'finished':
                    return int(status['returncode'])
                watchdog.observe(status)
                time.sleep(5)

        import ray

        ray.init(address=address)
        while True:
            live = [node for node in ray.nodes() if node['Alive'] and node['Resources'].get('GPU', 0) > 0]
            devices = sum(node['Resources']['GPU'] for node in live)
            writer.publish('starting', nodes=len(live), devices=devices)
            print(f'[multi-node] ready nodes={len(live)}/{nodes}, GPUs={devices}/{nodes * local_gpus}', flush=True)
            if len(live) == nodes and all(node['Resources']['GPU'] == local_gpus for node in live):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Not all Ray nodes/devices registered; refusing partial training')
            time.sleep(5)
        ray.shutdown()  # Disconnect only; leave this explicitly started cluster running.
        print('[multi-node] launching exactly one training driver', flush=True)
        child = subprocess.Popen(command)
        while child.poll() is None:
            writer.publish('training')
            time.sleep(5)
        result = child.returncode
        if result == 0 and os.environ.get('PA_OPD_VERIFY_FINAL_CHECKPOINT') == '1':
            result = 1
            verify_final_checkpoint(Path(os.environ['OUTPUT_DIR']), nodes * local_gpus)
            result = 0
        return result
    finally:
        # A supervisor failure must not leave a training driver running while
        # the other node exits. Normal completion never enters this branch.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        if owns_status:
            # Publish completion before the head pod exits. Bounded retries
            # also cover transient failure during the final status update.
            for attempt in range(12):
                if write_status(status_path, 'finished', returncode=result):
                    break
                time.sleep(2)
            else:
                raise RuntimeError('Unable to publish final supervisor status')


if __name__ == '__main__':
    sys.exit(main())

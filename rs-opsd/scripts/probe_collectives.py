#!/usr/bin/env python3
"""Bounded two-device torch/vLLM collective stress test; no model or dataset loading."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(iterations):
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    rank = int(os.environ['RANK'])
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=60))
    cpu_group = dist.new_group(backend='gloo', timeout=datetime.timedelta(seconds=60))
    comm = PyNcclCommunicator(cpu_group, device=device)
    if comm.disabled:
        raise RuntimeError('PyNcclCommunicator is disabled; test would not exercise the failing path')
    print('PROBE_INIT ' + json.dumps(dict(rank=rank, torch=torch.__version__,
          device=torch.cuda.get_device_name(device))), flush=True)
    x = torch.full((256,), rank + 1., device='cuda')
    gathered = torch.empty(512, device='cuda')
    output = torch.empty_like(x)
    started = time.monotonic()
    index = -1
    try:
        for index in range(iterations):
            # Training weight gather and vLLM PyNCCL use different communicators.
            dist.all_gather_into_tensor(gathered, x)
            comm.all_reduce(x, output)
            if (index + 1) % 128 == 0:
                torch.cuda.synchronize()
            if (index + 1) % 2048 == 0 or index + 1 == iterations:
                torch.cuda.synchronize()
                assert output.eq(3).all().item()
                assert gathered[:256].eq(1).all().item()
                assert gathered[256:].eq(2).all().item()
                print('PROBE_PROGRESS ' + json.dumps(dict(rank=rank, iterations=index+1,
                      collectives=2*(index+1), seconds=time.monotonic()-started)), flush=True)
    except Exception:
        print(f'PROBE_FAILED rank={rank} iteration={index+1}', flush=True)
        raise
    dist.destroy_process_group()
    print(f'PROBE_PASSED rank={rank}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--iterations', type=int, default=40000)
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()
    if args.iterations <= 0 or args.timeout <= 0:
        parser.error('iterations and timeout must be positive')
    if args.worker:
        worker(args.iterations)
        return
    results = {}
    for mode in ('current_environment',):
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        env.setdefault('NCCL_DEBUG', 'WARN')
        env.setdefault('OMP_NUM_THREADS', '1')
        # Show underlying allocation errors in job stdout, not a hidden node log.
        env.pop('NCCL_DEBUG_FILE', None)
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(Path(__file__).resolve()), '--worker',
               '--iterations', str(args.iterations)]
        print(f'PROBE_CASE_START mode={mode}', flush=True)
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            results[mode] = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # Only terminate the exact process group created for this probe.
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            results[mode] = 'timeout'
        print(f'PROBE_CASE_END mode={mode} result={results[mode]}', flush=True)
    print('PROBE_SUMMARY ' + json.dumps(results), flush=True)
    if results.get('current_environment') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

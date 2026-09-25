#!/usr/bin/env python3
"""Bounded TP checks: two-PPU generation or configurable single-node Ray groups."""

import argparse
import datetime
import inspect
import os
from pathlib import Path
import subprocess
import sys


def collective_probe():
    import torch
    import torch.distributed as dist
    from vllm.distributed import parallel_state as ps
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"PROBE rank={rank} current_device={torch.cuda.current_device()} "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"disable_pynccl={os.environ.get('VLLM_DISABLE_PYNCCL')}", flush=True)
    ps.init_distributed_environment(
        world_size=2, rank=rank, distributed_init_method="env://",
        local_rank=int(os.environ["LOCAL_RANK"]), backend="nccl",
        timeout=datetime.timedelta(seconds=90),
    )
    probe = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(probe)
    assert probe.item() == 3.0, probe
    print(f"TORCH_NCCL_OK rank={rank}", flush=True)
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=2, disable_custom_all_reduce=True))
    ps.set_custom_all_reduce(False)
    with set_current_vllm_config(config):
        ps.initialize_model_parallel(tensor_model_parallel_size=2)
        group = ps.get_tp_group()
        result = group.all_reduce(torch.tensor([rank + 1.0], device="cuda"))
        assert result.item() == 3.0, result
        gathered = group.all_gather(torch.tensor([rank + 1.0], device="cuda"), dim=0)
        assert gathered.tolist() == [1.0, 2.0], gathered
        torch.cuda.synchronize()
        print(f"VLLM_TP_COLLECTIVES_OK rank={rank}", flush=True)
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def model_probe():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.environ["MODEL_PATH"], tensor_parallel_size=2,
        dtype="bfloat16", trust_remote_code=True, enforce_eager=True,
        disable_custom_all_reduce=True, gpu_memory_utilization=0.45,
        max_model_len=512, max_num_batched_tokens=512, max_num_seqs=1,
    )
    outputs = llm.generate(["Reply with exactly the word: ok"],
                           SamplingParams(temperature=0.0, max_tokens=8))
    assert outputs and outputs[0].outputs and outputs[0].outputs[0].token_ids
    print(f"VLLM_TP_MODEL_OK output={outputs[0].outputs[0].text!r}", flush=True)


def ray_probe():
    """Reproduce the hybrid worker's masked device and existing mixed backend PG."""
    import ray
    import socket
    world_size = int(os.environ.get("PA_OPD_NUM_GPUS", "2"))
    tp_size = int(os.environ.get("PA_OPD_ROLLOUT_TP", "2"))
    assert world_size % tp_size == 0
    ray.init(num_gpus=world_size, include_dashboard=False, _temp_dir=f"/tmp/pa-tp-{os.getpid()}")

    @ray.remote(num_gpus=1)
    class Worker:
        def run(self, rank, address):
            import torch
            import torch.distributed as dist
            from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
            from vllm.distributed import parallel_state as ps
            torch.cuda.set_device(0)
            os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK="0")
            print(f"RAY_TP_DEVICE rank={rank} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                  f"assigned={ray.get_gpu_ids()} current={torch.cuda.current_device()}", flush=True)
            dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size,
                                    init_method=address, timeout=datetime.timedelta(seconds=90))
            t = torch.tensor([rank + 1.0], device="cuda")
            dist.all_reduce(t)
            assert t.item() == world_size * (world_size + 1) / 2
            print(f"RAY_TORCH_NCCL_OK rank={rank}", flush=True)
            config = VllmConfig(parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size, disable_custom_all_reduce=True,
                distributed_executor_backend="external_launcher"))
            ps.set_custom_all_reduce(False)
            with set_current_vllm_config(config):
                ps.init_distributed_environment(world_size=world_size, rank=rank, local_rank=0,
                                                distributed_init_method=address, backend="nccl")
                ps.initialize_model_parallel(tensor_model_parallel_size=tp_size)
                out = ps.get_tp_group().all_reduce(torch.tensor([rank + 1.0], device="cuda"))
                base = rank // tp_size * tp_size
                assert out.item() == sum(range(base + 1, base + tp_size + 1))
                print(f"RAY_VLLM_TP_OK rank={rank}", flush=True)
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()
            return rank

    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
    address = f"tcp://{ray.util.get_node_ip_address()}:{port}"
    workers = [Worker.remote() for _ in range(world_size)]
    try:
        assert ray.get([w.run.remote(i, address) for i, w in enumerate(workers)], timeout=180) == list(range(world_size))
        print("RAY_TP_PREFLIGHT_PASSED", flush=True)
    finally:
        for worker in workers:
            ray.kill(worker)
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["collective", "model", "ray"],
                        default=os.environ.get("PPU_TP_PROBE_PHASE"))
    args = parser.parse_args()
    if args.phase == "collective":
        collective_probe()
        return
    if args.phase == "model":
        model_probe()
        return
    if args.phase == "ray":
        ray_probe()
        return

    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    print("VLLM all_reduce implementation:\n" + inspect.getsource(CudaCommunicator.all_reduce), flush=True)
    for disable in ("0", "1"):
        env = dict(os.environ, VLLM_DISABLE_PYNCCL=disable, NCCL_DEBUG="INFO",
                   PYTHONUNBUFFERED="1", VLLM_LOGGING_LEVEL="INFO")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", str(Path(__file__).resolve()), "--phase", "collective"]
        print(f"START_COLLECTIVE disable_pynccl={disable}", flush=True)
        try:
            result = subprocess.run(cmd, env=env, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Collective probe timed out; do not start another GPU probe")
        print(f"END_COLLECTIVE disable_pynccl={disable} returncode={result.returncode}", flush=True)
        if result.returncode == 0:
            print(f"START_MODEL disable_pynccl={disable}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", "model"],
                           env=env, timeout=360, check=True)
            print(f"PPU_TP_PREFLIGHT_PASSED disable_pynccl={disable}", flush=True)
            return
    raise RuntimeError("Both native PyNCCL and fallback collective probes failed")


if __name__ == "__main__":
    main()

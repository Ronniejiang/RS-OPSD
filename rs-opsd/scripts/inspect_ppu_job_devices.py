#!/usr/bin/env python3
"""Read-only device summary; buffer nvidia-smi before Fuyao returns stdout."""
import csv
import json
import subprocess


def query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def main():
    devices = query("index,uuid,memory.used,memory.total")
    apps = query("gpu_uuid,pid,used_memory", "compute-apps")
    device_summary = {"devices": [
        {"id": d[0], "used_MiB": d[2], "total_MiB": d[3],
         "processes": [a[1:] for a in apps if a[0] == d[1]]}
        for d in devices]}
    # /proc/<pid>/environ does not reliably reflect Ray's runtime updates to
    # CUDA_VISIBLE_DEVICES. Use observed device/process allocation instead.
    print(json.dumps(device_summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

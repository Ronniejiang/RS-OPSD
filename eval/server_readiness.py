"""Choose a free serving port and verify the expected model before evaluation."""
import argparse
import json
import os
import socket
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def available_port(host: str, port: int) -> int:
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    # Do not set SO_REUSEPORT: reject listeners even when vLLM would share them.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
        return sock.getsockname()[1]


def verify_model(payload: dict, model_id: str) -> None:
    models = [item.get("id") for item in payload.get("data", [])]
    if models != [model_id]:
        raise ValueError(f"Wrong model endpoint: expected only {model_id!r}, received {models!r}")


def wait_for_model(api_base: str, model_id: str, timeout: float, pid: int | None = None) -> None:
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    successes = 0
    while time.monotonic() < deadline:
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError(
                    f"Model server process {pid} exited before becoming ready."
                ) from error
        try:
            with opener.open(api_base.rstrip("/") + "/models", timeout=5) as response:
                payload = json.load(response)
        except (URLError, TimeoutError, ConnectionError):
            successes = 0
        else:
            # Another model is a configuration error, not a slow startup.
            verify_model(payload, model_id)
            successes += 1
            if successes >= 3:
                print(f"Verified model {model_id} at {api_base}", flush=True)
                return
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Model {model_id!r} was not ready at {api_base} within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    port_parser = commands.add_parser("port")
    port_parser.add_argument("--host", default="127.0.0.1")
    port_parser.add_argument("--port", type=int, default=0)
    wait_parser = commands.add_parser("wait")
    wait_parser.add_argument("--api-base", required=True)
    wait_parser.add_argument("--model-id", required=True)
    wait_parser.add_argument("--timeout", type=float, default=600)
    wait_parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.command == "port":
        print(available_port(args.host, args.port))
    else:
        wait_for_model(args.api_base, args.model_id, args.timeout, args.pid)


if __name__ == "__main__":
    main()

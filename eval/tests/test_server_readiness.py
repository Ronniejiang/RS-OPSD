import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from eval.server_readiness import available_port, wait_for_model


class ReadinessTests(unittest.TestCase):
    def test_occupied_port_is_rejected(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            with self.assertRaises(OSError):
                available_port("127.0.0.1", sock.getsockname()[1])

    def test_model_identity_and_proxy_bypass(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "expected-model"}]}).encode())
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}):
                wait_for_model(url, "expected-model", 5)
                with self.assertRaisesRegex(ValueError, "Wrong model endpoint"):
                    wait_for_model(url, "other-model", 5)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_unavailable_server_times_out(self):
        port = available_port("127.0.0.1", 0)
        with self.assertRaises(TimeoutError):
            wait_for_model(f"http://127.0.0.1:{port}/v1", "expected", 0.1)


if __name__ == "__main__":
    unittest.main()

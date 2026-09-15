from __future__ import annotations

import io
import socket
import threading
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

SUSPECTS_CSV = "image_path,suspect_level\nimg_0.png,0.9\n"
SUSPECT_LEVEL_COUNTS_CSV = "suspect_level,count\n0.9,1\n"
WARNINGS_AND_ERRORS_CSV = "image_path,status\nimg_1.png,MISSING_IMAGE\n"


def _build_results_zip() -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("mislabel_suspects.csv", SUSPECTS_CSV)
        archive.writestr(
            "mislabel_suspect_level_counts.csv",
            SUSPECT_LEVEL_COUNTS_CSV,
        )
        archive.writestr("warnings_and_errors.csv", WARNINGS_AND_ERRORS_CSV)
    return archive_buffer.getvalue()


RESULTS_ZIP = _build_results_zip()


class LocalTransportServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), LocalTransportRequestHandler)
        self.request_counts: Counter[str] = Counter()
        self.request_headers: dict[str, list[dict[str, str]]] = {}
        self.disconnect_sse = threading.Event()
        self.release_slow_response = threading.Event()

    @property
    def base_url(self) -> str:
        address = self.server_address[0]
        port = self.server_address[1]
        return f"http://{address}:{port}"

    def record_request(self, path: str, headers: dict[str, str]) -> int:
        self.request_counts[path] += 1
        self.request_headers.setdefault(path, []).append(headers)
        return self.request_counts[path]

    def release_handlers(self) -> None:
        self.disconnect_sse.set()
        self.release_slow_response.set()


class LocalTransportRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def transport_server(self) -> LocalTransportServer:
        return cast("LocalTransportServer", self.server)

    def do_GET(self) -> None:  # noqa: N802
        request_number = self.transport_server.record_request(
            self.path,
            {name: value for name, value in self.headers.items()},
        )

        if self.path == "/results.zip":
            self._send_bytes("application/zip", RESULTS_ZIP)
        elif self.path == "/events":
            self._send_sse(request_number)
        elif self.path == "/slow":
            self._send_slow_response()
        elif self.path == "/retry-success":
            self._send_retry_response(request_number, failures=2)
        elif self.path == "/retry-exhausted":
            self._send_retry_response(request_number, failures=10)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return None

    def _send_bytes(self, content_type: str, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_sse(self, request_number: int) -> None:
        if request_number == 1:
            content = (
                b"id: event-1\nevent: progress\nretry: 0\ndata: first\ndata: line\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content) + 1))
            self.end_headers()
            self.wfile.write(content)
            self.wfile.flush()
            self.transport_server.disconnect_sse.wait(timeout=1.0)
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return

        content = b"id: event-2\nevent: complete\ndata: done\n\n"
        self._send_bytes("text/event-stream", content)

    def _send_slow_response(self) -> None:
        self.transport_server.release_slow_response.wait(timeout=1.0)
        try:
            self._send_bytes("text/plain", b"late")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def _send_retry_response(self, request_number: int, failures: int) -> None:
        if request_number <= failures:
            content = b'{"detail":"try again"}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_bytes("application/json", b'{"status":"ok"}')

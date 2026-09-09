from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from tests.transport.server import LocalTransportServer

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def local_transport_server() -> Generator[LocalTransportServer, None, None]:
    server = LocalTransportServer()
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="local-transport-server",
        daemon=True,
    )
    server_thread.start()
    try:
        yield server
    finally:
        server.release_handlers()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        if server_thread.is_alive():
            raise RuntimeError("Local transport server did not stop")

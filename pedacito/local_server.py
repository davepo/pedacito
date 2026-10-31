"""A local ``ThreadingHTTPServer`` that does not perform reverse DNS.

``http.server.HTTPServer.server_bind()`` sets ``server_name`` with
``socket.getfqdn(host)``.  For a loopback bind that reverse lookup is pure
overhead, and on macOS (mDNS) it can block for tens of seconds -- long enough
for the review page and tests that wait for a server to start to time out.
We already know the bind address, so use it directly.
"""

from __future__ import annotations

import socketserver
from http.server import ThreadingHTTPServer


class LocalHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer bound without a reverse-DNS lookup."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port

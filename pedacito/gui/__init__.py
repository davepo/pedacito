"""The browser control panel: `pedacito gui`.

    pedacito gui                    # opens http://127.0.0.1:<port>/ in a browser
    pedacito gui --port 8756        # fixed port instead of an OS-chosen one
    pedacito gui --no-open          # print the URL instead of opening a browser

A single-page app over a local, dependency-free server (stdlib only, no CDN, so
it works offline). Layout: chat on the left, proposed changes (old | updated,
with accept/reject) and a live terminal in the middle, project files on the
right. Streaming uses Server-Sent Events.

Modules:
    server.py        routes, static files, the event stream
    core.py          shared state: event bus, chat transcript, staged review
    jobs.py          background jobs (open, index, task, explain, auto, new)
    settings.py      comment-preserving config edits (pure, unit-tested)
    settings_api.py  config files on disk: backups, secrets, test connection
    static/          index.html, app.css and the scripts
"""

from .server import create_server, run

__all__ = ["create_server", "run"]

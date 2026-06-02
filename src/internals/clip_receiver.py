#!/usr/bin/env python3
"""Standalone laptop receiver for wall-camera data-collection clips.

NON-PRODUCTION dev tool. Stdlib only — run it directly, no backend, no
deps:

    python3 src/internals/clip_receiver.py --out ./collected_clips --port 8765

It accepts the exact multipart upload the wall camera's `ClipUploader`
sends (POST <base>/upload, one part named "clip", Content-Type
video/mp4, plus an X-Camera-Identity header) and writes each clip as a
plain, renamable .mp4 into the --out folder. Point the camera's
"Experiment Server URL" setting at http://<this-laptop-lan-ip>:<port>.

Files land flat (e.g. collected_clips/clip_20260602-141530.mp4) so they
are trivial to rename/sort for training-data curation. The camera
identity is only woven into the name when needed to avoid a collision.
"""

import argparse
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Module globals set in main(); the stdlib handler API gives us no clean
# per-instance injection point, so a module-level config is simplest.
OUT_DIR: Path = Path("collected_clips")

_BOUNDARY_RE = re.compile(rb'boundary=("?)([^";]+)\1')
_FILENAME_RE = re.compile(rb'filename="([^"]*)"')
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(name: str) -> str:
    """Collapse anything not filename-safe to '_'."""
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "clip"


def _unique_path(out_dir: Path, base_name: str, identity: str | None) -> Path:
    """A non-colliding path in `out_dir`. Keeps the folder flat; only
    weaves in the camera identity / a counter when the plain name is
    already taken.
    """
    candidate = out_dir / base_name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix or ".mp4"
    tag = _safe(identity)[-8:] if identity else ""
    if tag:
        candidate = out_dir / f"{stem}_{tag}{suffix}"
        if not candidate.exists():
            return candidate
    n = 1
    while True:
        candidate = out_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _extract_clip(body: bytes, boundary: bytes) -> tuple[bytes, str | None]:
    """Pull the single `clip` part out of a multipart/form-data body.

    Returns (file_bytes, filename). Splitting on the boundary bytes is
    safe for binary payloads: the boundary carries a per-upload UUID, so
    it won't occur inside the MP4. Matches the envelope built by
    ClipUploader.buildMultipartBody (one part, name="clip").
    """
    delimiter = b"--" + boundary
    sections = body.split(delimiter)
    for section in sections:
        # Skip the preamble ("") and the closing "--\r\n" marker.
        if not section or section in (b"--\r\n", b"--", b"\r\n"):
            continue
        # A part is: \r\n <headers> \r\n\r\n <data> \r\n
        part = section[2:] if section.startswith(b"\r\n") else section
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers = part[:header_end]
        if b'name="clip"' not in headers:
            continue
        data = part[header_end + 4:]
        if data.endswith(b"\r\n"):
            data = data[:-2]
        fn_match = _FILENAME_RE.search(headers)
        filename = fn_match.group(1).decode("utf-8", "replace") if fn_match else None
        return data, filename
    return b"", None


class ClipHandler(BaseHTTPRequestHandler):
    server_version = "NemaraClipReceiver/1.0"

    def _reply(self, code: int, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # stdlib BaseHTTPRequestHandler naming
        # Lets you sanity-check reachability from a browser / curl.
        if self.path in ("/", "/health"):
            self._reply(200, "nemara clip receiver: POST a clip to /upload\n")
        else:
            self._reply(404, "not found\n")

    def do_POST(self) -> None:  # stdlib BaseHTTPRequestHandler naming
        if self.path.rstrip("/") != "/upload":
            self._reply(404, "not found (POST /upload)\n")
            return

        ctype = self.headers.get("Content-Type", "")
        m = _BOUNDARY_RE.search(ctype.encode("utf-8"))
        if "multipart/form-data" not in ctype or not m:
            self._reply(400, "expected multipart/form-data with a boundary\n")
            return
        boundary = m.group(2)

        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self._reply(411, "missing/zero Content-Length\n")
            return
        body = self.rfile.read(length)

        data, filename = _extract_clip(body, boundary)
        if not data:
            self._reply(422, "no 'clip' part found in body\n")
            return

        identity = self.headers.get("X-Camera-Identity")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = _safe(filename) if filename else f"clip_{stamp}.mp4"
        if not base_name.lower().endswith(".mp4"):
            base_name += ".mp4"

        dest = _unique_path(OUT_DIR, base_name, identity)
        dest.write_bytes(data)

        size_kb = len(data) / 1024
        line = f"saved {dest}  ({size_kb:.0f} KB, identity={identity or '-'})"
        print(line, flush=True)
        self._reply(200, line + "\n")

    def log_message(self, fmt: str, *args) -> None:
        # Quieter default access logging; our own print() is the signal.
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wall-camera clip receiver (dev).")
    parser.add_argument(
        "--out", default="collected_clips",
        help="Directory to write clips into (default: ./collected_clips)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0 — reachable on the LAN)",
    )
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out).expanduser().resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), ClipHandler)
    print(f"listening on :{args.port} → {OUT_DIR}", flush=True)
    print("point the camera's Experiment Server URL at this host, then press Capture.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Regenerate the compressed test fixture: sample.pgn -> sample.pgn.zst.

Both files are committed (via .gitignore exceptions) so the ingest smoke test runs for everyone
without the 30 GB monthly download. Rerun this after editing sample.pgn:

    python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import zstandard

HERE = Path(__file__).resolve().parent


def main() -> None:
    src = HERE / "sample.pgn"
    dst = HERE / "sample.pgn.zst"
    cctx = zstandard.ZstdCompressor(level=19)  # deterministic given input + level
    dst.write_bytes(cctx.compress(src.read_bytes()))
    print(f"wrote {dst} ({dst.stat().st_size} bytes) from {src.name}")


if __name__ == "__main__":
    main()

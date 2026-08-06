#!/usr/bin/env python3
"""End-to-end regression test for MTP checkpoint restore followed by new media.

Requires the production-compatible Qwen MTP server on port 8001 and a Pi session
containing multimodal history. The first continuation seeds the prompt cache; the
second appends a new image and must complete without restarting llama-server.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def health(endpoint: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except Exception:
            time.sleep(1)
    raise AssertionError(f"server did not become healthy within {timeout:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--pi", default="/home/chihmin/.local/bin/pi")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    if not args.session.is_file() or not args.image.is_file():
        parser.error("session and image must both exist")

    health(args.endpoint)
    with tempfile.TemporaryDirectory(prefix="mtp-mm-replay-") as tmp:
        replay = Path(tmp) / "session.jsonl"
        shutil.copy2(args.session, replay)
        base = [
            args.pi,
            "--session", str(replay),
            "--provider", "local-llama",
            "--model", "qwen3.6-35b-q4",
            "--no-tools", "--no-skills", "--no-extensions", "--no-context-files",
            "--thinking", "off",
        ]

        seed = subprocess.run(
            base + ["-p", "Diagnostic cache seed: reply exactly OK."],
            text=True, capture_output=True, timeout=180,
        )
        assert seed.returncode == 0, seed.stderr or seed.stdout

        pid_before = subprocess.check_output(
            ["systemctl", "show", "-p", "MainPID", "--value", "qwen-mtp.service"],
            text=True,
        ).strip()

        journal_since = datetime.now().astimezone().isoformat()
        append = subprocess.run(
            base + ["-p", f"@{args.image}", "Diagnostic appended image: reply exactly OK."],
            text=True, capture_output=True, timeout=180,
        )
        assert append.returncode == 0, append.stderr or append.stdout

        journal = subprocess.check_output(
            [
                "journalctl", "-u", "qwen-mtp.service", "--since", journal_since,
                "--no-pager", "-o", "cat",
            ],
            text=True,
        )
        generated = [
            int(match.group(1))
            for match in re.finditer(r"#gen drafts = (\d+)", journal)
        ]
        assert any(count > 0 for count in generated), (
            "multimodal continuation completed without MTP drafts\n" + journal
        )

        pid_after = subprocess.check_output(
            ["systemctl", "show", "-p", "MainPID", "--value", "qwen-mtp.service"],
            text=True,
        ).strip()
        assert pid_after == pid_before, f"llama-server restarted: {pid_before} -> {pid_after}"

    print("PASS: MTP checkpoint restore accepted newly appended media without restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

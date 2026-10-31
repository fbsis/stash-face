#!/usr/bin/env python3
"""Run real application code against a local Postman Stash mock. No cloud writes."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    port = int(os.getenv("STASH_MOCK_PORT", "49171"))
    url = f"http://127.0.0.1:{port}/stash"
    spec = ROOT / "postman/simulations/review.sim.yaml"
    with tempfile.NamedTemporaryFile(prefix="stash-review-simulation-", suffix=".ndjson", delete=False) as log:
        process = subprocess.Popen(["postman", "simulation", "run", str(spec), "--port", str(port),
            "--no-history", "--output", "ndjson"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            with httpx.Client(timeout=0.5, trust_env=False) as client:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError(Path(log.name).read_text())
                    try:
                        if client.get(url + "/test/state").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Postman simulation did not become ready")
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "integration"],
                                    cwd=ROOT, env=os.environ | {"STASH_MOCK_URL": url})
        finally:
            os.killpg(process.pid, signal.SIGINT) if process.poll() is None else None
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        events = []
        for line in Path(log.name).read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "request":
                events.append(event)
        print(f"Simulation log: {log.name} ({len(events)} recorded requests)")
        paths = {event.get("path") for event in events if event.get("statusCode") == 200}
        if not {"/graphql", "/image/1"}.issubset(paths):
            raise RuntimeError("No successful GraphQL/image request evidence; integration is not verified")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

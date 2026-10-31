"""Standard-library-only Stash dispatcher. Detection never runs on the Stash VM."""
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def main(payload):
    args = payload.get("args") or {}
    hook = args.get("hookContext") or {}
    config_path = Path(__file__).with_name("stash_face.json")
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    endpoint = (os.getenv("FACE_SERVICE_URL") or config.get("service_url", "")).rstrip("/")
    token = os.getenv("FACE_TOKEN") or config.get("token", "")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or len(token) < 32
            or token.startswith(("replace-", "change-"))):
        raise ValueError("Configure a valid service_url and random token in stash_face.json or environment")
    client = build_opener(NoRedirects())

    def call(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = Request(endpoint + path, data=body, headers={
            "X-API-Token": token, "Content-Type": "application/json"})
        with client.open(req, timeout=3 if hook else 15) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError("Remote response exceeds the size limit")
            return json.loads(raw)

    action = args.get("action", "scan")
    if action == "health":
        result = {"service": call("/health"), "stash": call("/stash/status"), "dashboard": endpoint}
    elif action in ("scan", "image", "forget", "cluster", "sync_performers"):
        body = {"action": action}
        if action in ("image", "forget"):
            image_id = str(hook.get("id") or args.get("image_id") or "")
            if not image_id.isdecimal() or int(image_id) <= 0:
                raise ValueError("An image ID is required")
            body["image_id"] = image_id
        result = call("/jobs", body)
    else:
        raise ValueError("Unsupported action; Performer assignment requires explicit review in the dashboard")
    return {"output": result}


if __name__ == "__main__":
    payload = {}
    try:
        raw = sys.stdin.read(65537)
        if len(raw) > 65536:
            raise ValueError("Input exceeds the size limit")
        payload = json.loads(raw)
        print(json.dumps(main(payload)))
    except (HTTPError, URLError, TimeoutError, OSError):
        hook = ((payload.get("args") or {}).get("hookContext") or {})
        if hook:
            print(json.dumps({"output": {"queued": False,
                "reason": "Remote service unavailable; run a scan after it recovers"}}))
            sys.exit(0)  # never block native Stash edits because the remote worker is offline
        print(json.dumps({"error": "Remote service connection failed; check URL, token, TLS and firewall"}))
        sys.exit(1)
    except Exception:
        print(json.dumps({"error": "Invalid plugin input or configuration; check the example configuration"}))
        sys.exit(1)

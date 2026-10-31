#!/usr/bin/env python3
import json
from pathlib import Path

from stash_face.app import create_app

path = Path(__file__).resolve().parents[1] / "docs/openapi.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
print(path)

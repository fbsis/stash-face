"""Benchmark bounding-box detection, never identity matching."""
import argparse
import json
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from .config import Settings
from .detector import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["opencv", "openvino"], default="opencv")
    parser.add_argument("--device", choices=["CPU", "GPU", "AUTO"], default="CPU")
    parser.add_argument("--model", default="")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB") if args.image else Image.new(
        "RGB", (640, 480), "white")
    with tempfile.TemporaryDirectory() as directory:
        base = Settings(stash_url="http://127.0.0.1:9999", token="benchmark-only-" + "x" * 32,
                        data_dir=Path(directory))
        detector = Detector(replace(base, provider=args.provider, device=args.device,
                                    cpu_fallback=False, model_path=args.model))
        detector.detect(image)
        durations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            faces = detector.detect(image)
            durations.append((time.perf_counter() - start) * 1000)
        print(json.dumps({"detector": detector.info(), "median_ms": statistics.median(durations),
                          "iterations": len(durations), "face_count": len(faces),
                          "synthetic_blank_input": args.image is None,
                          "warning": "A blank-image timing is not a real-world throughput benchmark."}))


if __name__ == "__main__":
    main()

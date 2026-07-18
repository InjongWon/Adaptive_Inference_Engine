"""Capture the raw vLLM /metrics endpoint before and after experiments."""

import argparse
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/metrics")
    parser.add_argument("--output", default="results/vllm_metrics.prom")
    args = parser.parse_args()

    response = httpx.get(args.url, timeout=10.0)
    response.raise_for_status()
    Path(args.output).write_text(response.text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

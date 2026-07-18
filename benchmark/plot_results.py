import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--metric", default="output_tokens_per_second")
    parser.add_argument("--output", default="results/comparison.png")
    args = parser.parse_args()

    labels, values = [], []
    for filename in args.files:
        data = json.loads(Path(filename).read_text())
        labels.append(data.get("workload", Path(filename).stem))
        values.append(data["summary"].get(args.metric) or 0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel(args.metric.replace("_", " "))
    plt.title("LLM inference benchmark comparison")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

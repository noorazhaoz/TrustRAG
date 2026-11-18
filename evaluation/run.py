
"""Command-line entry point for TrustRAG evaluation."""

import argparse

from evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    completed = run_evaluation(
        input_path=args.input,
        output_path=args.output,
        api_url=args.api_url,
        max_tokens=args.max_tokens,
    )

    print(f"Completed {completed} evaluation questions.")


if __name__ == "__main__":
    main()

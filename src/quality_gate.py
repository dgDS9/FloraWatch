import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics_file",
        type=str,
        default="models/evaluation_metrics.json",
    )

    parser.add_argument(
        "--min_val_accuracy",
        type=float,
        default=0.86,
    )

    parser.add_argument(
        "--min_val_top3",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)

    if not metrics_path.exists():
        print(f"❌ Quality Gate failed: {metrics_path} not found.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    val_accuracy = metrics["finetune_val_accuracy"]
    val_top3 = metrics["finetune_val_top3"]

    print("\n=== FloraWatch Quality Gate ===")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Required Accuracy:   {args.min_val_accuracy:.4f}")
    print()
    print(f"Validation Top-3:    {val_top3:.4f}")
    print(f"Required Top-3:      {args.min_val_top3:.4f}")

    accuracy_passed = val_accuracy >= args.min_val_accuracy
    top3_passed = val_top3 >= args.min_val_top3

    if accuracy_passed and top3_passed:
        print("\n✅ QUALITY GATE PASSED")
        sys.exit(0)

    print("\n❌ QUALITY GATE FAILED")

    if not accuracy_passed:
        print(
            f"- Validation Accuracy too low: "
            f"{val_accuracy:.4f} < {args.min_val_accuracy:.4f}"
        )

    if not top3_passed:
        print(
            f"- Validation Top-3 too low: "
            f"{val_top3:.4f} < {args.min_val_top3:.4f}"
        )

    sys.exit(1)


if __name__ == "__main__":
    main()
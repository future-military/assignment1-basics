import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("logs/training_curves.png"),
    )

    return parser.parse_args()


def read_log(
    log_path: Path,
) -> dict[str, list[float]]:
    steps = []
    train_losses = []
    validation_steps = []
    validation_losses = []
    learning_rates = []
    tokens_per_second = []

    with open(
        log_path,
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])

            steps.append(step)
            train_losses.append(
                float(row["train_loss"])
            )
            learning_rates.append(
                float(row["learning_rate"])
            )
            tokens_per_second.append(
                float(row["tokens_per_second"])
            )

            validation_value = row[
                "validation_loss"
            ].strip()

            if validation_value:
                validation_steps.append(step)
                validation_losses.append(
                    float(validation_value)
                )

    if not steps:
        raise ValueError(
            f"No training rows found in {log_path}"
        )

    return {
        "steps": steps,
        "train_losses": train_losses,
        "validation_steps": validation_steps,
        "validation_losses": validation_losses,
        "learning_rates": learning_rates,
        "tokens_per_second": tokens_per_second,
    }


def plot_log(
    data: dict[str, list[float]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 10),
        sharex=True,
    )

    loss_axis = axes[0]
    learning_rate_axis = axes[1]
    throughput_axis = axes[2]

    loss_axis.plot(
        data["steps"],
        data["train_losses"],
        label="Training loss",
        color="tab:blue",
        alpha=0.8,
    )

    if data["validation_steps"]:
        loss_axis.plot(
            data["validation_steps"],
            data["validation_losses"],
            label="Validation loss",
            color="tab:orange",
            marker="o",
        )

    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Training and validation loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    learning_rate_axis.plot(
        data["steps"],
        data["learning_rates"],
        color="tab:green",
    )
    learning_rate_axis.set_ylabel("Learning rate")
    learning_rate_axis.set_title("Learning-rate schedule")
    learning_rate_axis.grid(alpha=0.3)

    throughput_axis.plot(
        data["steps"],
        data["tokens_per_second"],
        color="tab:red",
    )
    throughput_axis.set_xlabel("Training step")
    throughput_axis.set_ylabel("Tokens / second")
    throughput_axis.set_title("Training throughput")
    throughput_axis.grid(alpha=0.3)

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_args()

    data = read_log(args.log_path)

    plot_log(
        data=data,
        output_path=args.output_path,
    )

    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
import mlflow
from mlflow import MlflowClient


MODEL_NAME = "FloraWatchClassifier"
METRIC = "finetune_val_accuracy"


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_registry_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger",
    )

    champion_run = client.get_run(champion.run_id)
    challenger_run = client.get_run(challenger.run_id)

    champion_score = champion_run.data.metrics[METRIC]
    challenger_score = challenger_run.data.metrics[METRIC]

    print("\n=== Champion vs Challenger ===")

    print(
        f"Champion v{champion.version}: "
        f"{champion_score:.4f}"
    )

    print(
        f"Challenger v{challenger.version}: "
        f"{challenger_score:.4f}"
    )

    if challenger_score > champion_score:

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            challenger.version,
        )

        print(
            f"\n✅ Challenger v{challenger.version} "
            f"is now Champion."
        )

    else:

        print(
            f"\n❌ Challenger is not better."
        )

        print(
            f"Champion remains v{champion.version}."
        )


if __name__ == "__main__":
    main()
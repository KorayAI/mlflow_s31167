import json

import mlflow
from mlflow import MlflowClient

from train import ROOT, URI


def main():
    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    winner = json.loads((ROOT / "reports/latest.json").read_text())
    if winner["validation_f1"] < 0.90:
        raise RuntimeError("Winner does not meet the F1 promotion threshold")
    version = mlflow.register_model("runs:/" + winner["run_id"] + "/model", "breast-cancer-clf")
    MlflowClient().set_registered_model_alias("breast-cancer-clf", "champion", version.version)
    print("Registered version", version.version, "with champion alias")


if __name__ == "__main__":
    main()

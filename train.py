from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unavailable", "dirty": "not a Git repository"}


def train(config, experiment, batch, source=None):
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    digest = hashlib.sha256(X.assign(target=y).to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    indices = np.arange(len(y))
    tr, va = train_test_split(indices, test_size=config["test_size"], random_state=config["seed"], stratify=y)
    split_hash = hashlib.sha256(json.dumps({"train": tr.tolist(), "validation": va.tolist()}, sort_keys=True).encode()).hexdigest()
    if source:
        old = MlflowClient().get_run(source).data.params
        if old["dataset_sha256"] != digest or old["split_sha256"] != split_hash:
            raise ValueError("Dataset or split does not match the source run")
    estimator = LogisticRegression(C=config["C"], max_iter=config["max_iter"], random_state=config["seed"]) if config["model"] == "logistic_regression" else RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=config["seed"], n_jobs=1)
    model = make_pipeline(StandardScaler(), estimator)
    with mlflow.start_run(run_name=config["model"]) as run:
        mlflow.set_tags({"batch_id": batch, "source_run_id": source or "", "git_commit": git_state()["commit"], "git_dirty": str(git_state()["dirty"])})
        mlflow.log_params({**config, "dataset_sha256": digest, "split_sha256": split_hash, "preprocessing": "StandardScaler fitted on training only", "positive_label": "0=malignant"})
        model.fit(X.iloc[tr], y.iloc[tr])
        metrics = {}
        for label, rows in [("train", tr), ("validation", va)]:
            pred = model.predict(X.iloc[rows])
            metrics[label + "_accuracy"] = accuracy_score(y.iloc[rows], pred)
            metrics[label + "_f1"] = f1_score(y.iloc[rows], pred, pos_label=0)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict({"dataset_sha256": digest, "split_sha256": split_hash, "git": git_state(), "python": sys.version, "train_indices": tr.tolist(), "validation_indices": va.tolist()}, "reproducibility.json")
        mlflow.log_text(pd.DataFrame({"row": va, "actual": y.iloc[va].to_numpy(), "prediction": model.predict(X.iloc[va])}).to_csv(index=False), "validation_predictions.csv")
        mlflow.log_dict({"labels": [0, 1], "matrix": confusion_matrix(y.iloc[va], model.predict(X.iloc[va]), labels=[0, 1]).tolist()}, "confusion_matrix.json")
        mlflow.log_artifact(str(ROOT / "requirements.txt"))
        mlflow.log_artifact(str(ROOT / "train.py"), "source")
        mlflow.sklearn.log_model(model, artifact_path="model", signature=infer_signature(X.iloc[tr], model.predict(X.iloc[tr])), input_example=X.iloc[tr[:5]], pip_requirements=str(ROOT / "requirements.txt"))
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="thesis-mlflow-demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--register-best", action="store_true")
    parser.add_argument("--model-name", default="breast-cancer-clf")
    parser.add_argument("--config-from-run")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("test-size must be between zero and one")
    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        client.create_experiment(args.experiment, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(args.experiment)
    batch = uuid.uuid4().hex
    if args.config_from_run:
        config_path = mlflow.artifacts.download_artifacts(run_id=args.config_from_run, artifact_path="config.json")
        configs = [json.loads(Path(config_path).read_text())]
    else:
        common = {"seed": args.seed, "test_size": args.test_size, "C": 1.0, "max_iter": 2000, "n_estimators": 100, "max_depth": 5}
        configs = [{**common, "model": name} for name in ["logistic_regression", "random_forest"]]
    ids = [train(c, args.experiment, batch, args.config_from_run) for c in configs]
    runs = [client.get_run(rid) for rid in ids]
    rows = [{"run_id": r.info.run_id, "model": r.data.params["model"], **r.data.metrics, "dataset_sha256": r.data.params["dataset_sha256"], "split_sha256": r.data.params["split_sha256"]} for r in runs]
    table = pd.DataFrame(rows).sort_values(["validation_f1", "validation_accuracy", "model"], ascending=[False, False, True])
    report = ROOT / "reports" / batch
    report.mkdir(parents=True)
    table.to_csv(report / "comparison.csv", index=False)
    winner = table.iloc[0].to_dict()
    winner["batch_id"] = batch
    winner["promotion_rule"] = "Highest validation malignant-class F1, then accuracy, then alphabetical model name; F1 must be >= 0.90. Selection is within this batch, not against prior champions."
    winner["rerun_command"] = "python train.py --config-from-run " + winner["run_id"]
    if args.register_best:
        if winner["validation_f1"] < 0.90:
            raise RuntimeError("No model meets the promotion threshold")
        version = mlflow.register_model("runs:/" + winner["run_id"] + "/model", args.model_name)
        client.set_registered_model_alias(args.model_name, "champion", version.version)
        resolved = client.get_model_version_by_alias(args.model_name, "champion")
        assert resolved.run_id == winner["run_id"]
        loaded = mlflow.sklearn.load_model("models:/" + args.model_name + "@champion")
        loaded.predict(load_breast_cancer(as_frame=True).data.iloc[:5])
        winner.update({"registered_model": args.model_name, "version": version.version, "alias": "champion", "alias_load_verified": True})
    write_json(report / "winner.json", winner)
    write_json(ROOT / "reports" / "latest.json", {"report": str(report.relative_to(ROOT)), **winner})
    print(table.to_string(index=False))
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()

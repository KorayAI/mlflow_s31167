import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train import ROOT, URI


def main():
    mlflow.set_tracking_uri(URI)
    evidence = json.loads((ROOT / "reports/latest.json").read_text())
    candidates = pd.read_csv(ROOT / evidence["report"] / "comparison.csv")
    ids = candidates.run_id.tolist()
    rows = []
    data = load_breast_cancer(as_frame=True)
    for rid in ids:
        run = mlflow.MlflowClient().get_run(rid)
        _, X_val, _, y_val = train_test_split(data.data, data.target, test_size=float(run.data.params["test_size"]), random_state=int(run.data.params["seed"]), stratify=data.target)
        model = mlflow.sklearn.load_model("runs:/" + rid + "/model")
        positive_column = list(model.classes_).index(0)
        auc = roc_auc_score((y_val == 0).astype(int), model.predict_proba(X_val)[:, positive_column])
        label = run.data.params["model"].replace("logistic_regression", "baseline-logreg").replace("random_forest", "rf-100-d5")
        rows.append({"run_name": label, "run_id": rid, "val_f1": run.data.metrics["validation_f1"], "val_roc_auc": auc})
    table = pd.DataFrame(rows).sort_values("val_f1", ascending=False, kind="stable")
    table.to_csv(ROOT / "reports/runs_comparison.csv", index=False)
    lines = ["# s31167 — validation comparison", "", "F1 and ROC-AUC both use malignant (0) as the positive class. ROC-AUC is computed from saved model probabilities on each run's original validation split. These are measured results, not values copied from the reference image.", "", "| Run | Validation F1 | Validation ROC-AUC |", "| --- | ---: | ---: |"]
    lines += [f"| {r.run_name} | {r.val_f1:.6f} | {r.val_roc_auc:.6f} |" for r in table.itertuples()]
    (ROOT / "reports/runs_comparison.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    positions = list(range(len(table)))
    ax.barh([y - 0.19 for y in positions], table.val_f1, height=0.38, color="#3d73ab", label="val F1")
    ax.barh([y + 0.19 for y in positions], table.val_roc_auc, height=0.38, color="#9bb8d5", label="val ROC-AUC")
    ax.set_yticks(positions, table.run_name)
    ax.invert_yaxis()
    ax.set_xlim(0.90, 1.0)
    ax.set_title("s31167 — validation metrics", fontsize=17, pad=15)
    ax.set_xlabel("Score (axis starts at 0.90)")
    ax.legend(loc="lower right")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/runs_comparison.png", dpi=180)
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

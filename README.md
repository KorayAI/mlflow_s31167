# s31167 — MLflow experiment tracking and model registry

Predict malignant (0) versus benign (1) tumors using the 569-row, 30-feature Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn. Attribution: Wolberg, Mangasarian, Street and Street (1993), [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI](https://doi.org/10.24432/C5DW2B). Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Educational use; this is not a clinical system.

## Install and run

Use Python 3.12. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --register-best
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5055
```

On Windows activate with `.venv\Scripts\Activate.ps1`. Open http://127.0.0.1:5055. The named experiment is `thesis-mlflow-demo`. Tracking and registry use SQLite at the absolute project path to `mlflow.db`; artifacts live in `mlartifacts/`. The script configures its own URI, independent of MLFLOW_TRACKING_URI. Training needs no dataset download.

## Runs, selection and registry

Compare logistic regression (C=1, max_iter=2000) with random forest (100 trees, max_depth=5). Both use StandardScaler fitted on training data only and the same stratified 75/25 split with seed 42. Each run logs params, training and validation accuracy/F1, a fitted pipeline with signature/input example, confusion matrix, predictions, source, requirements and reproducibility metadata. Malignant (0) is the positive F1 class.

Highest validation F1 wins, then accuracy, then alphabetical model name. Registration requires F1 >= 0.90 and assigns `breast-cancer-clf@champion` to the new version. Selection is within the current invocation; it can replace a prior champion. Validation scores are used for model selection, not an untouched final test estimate.

The supplied measured comparison shows logistic regression validation F1 0.981132 and accuracy 0.986014; forest F1 0.933333 and accuracy 0.951049. Logistic regression won on F1. The original winner was registered as version 1 with champion alias. A rerun reproduced identical parameters and metrics, and champion loading and prediction were verified.

## Evidence

`reports/comparison.csv` contains the executed candidate comparison. `reports/winner.json` identifies the original winner, registry version, alias, data/split hashes and exact rerun command. `screenshots/Comparison.png` and `screenshots/registry.png` are genuine captures from this project's MLflow 2.22 UI. No database or model binaries are included in this submission; execute training to create your own local store. Historical run IDs in supplied reports are not available in a new empty database.

## Reproduce from logged metadata

Every execution creates `reports/<batch_id>/comparison.csv`, `winner.json`, and `reports/latest.json`. Use the winning ID from your local report:

```bash
python train.py --config-from-run RUN_ID
```

Replace RUN_ID with the real ID. The script downloads config.json from that run, restores all explicit classifier settings, seed and split size, checks dataset/split hashes, and records a linked new run. Inspect params with `mlflow.MlflowClient().get_run(RUN_ID).data.params` after setting the tracking URI. Logged reproducibility.json contains exact row indices and Git state. The SHA-256 identifies a canonical CSV of the actual scikit-learn feature/target table, rather than the original UCI download.

The original run occurred before Git initialization and explicitly records unavailable commit metadata. Commit code before retraining to record a real commit ID. Direct requirements are pinned; transitive versions are not fully locked. Match Python/dependency versions and logged source for older runs. Exact numerical reproduction across other hardware/versions is not guaranteed.

## GitHub submission

Upload the supplied files, preserving the reports and screenshots folders. Keep .env, credentials, local databases, model artifact stores and virtual environments out of Git; .gitignore excludes them. Submit the repository URL.

```bash
git init
git add .
git commit -m "Add s31167 MLflow project and evidence"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mlflow_s31167.git
git push -u origin main
```

Replace YOUR_USERNAME with your GitHub username and create the empty repository first. GitHub upload and Windows/Linux execution were not performed. The training and registry workflow was tested on macOS/Python 3.12; this simplified package uses the same training source.

## Additional evidence

`reports/screenshots/` contains the experiment list, winning run overview, winning metrics, model registry, and parallel coordinates comparison. `reports/runs_comparison.csv`, `.md`, and `.png` include the s31167 chart and measured F1/ROC-AUC results for the two candidates and the winning reproduction. ROC-AUC was calculated afterward from saved model probabilities using the original validation split and malignant (0) as positive. The chart axis begins at 0.90. The top-level `screenshots/Comparison.png` and `screenshots/registry.png` are convenient copies of the main comparison and registry evidence.

## Separate comparison and registration commands

`train.py` supports comparison and registration itself. The following equivalent workflow uses separate scripts:

```bash
python train.py
python register.py
python compare.py
```

On macOS/Linux use `bash run_experiments.sh` to execute these three commands. `register.py` reads the latest local winning run, enforces F1 >= 0.90, registers a new model version and assigns champion. `compare.py` exports the latest batch as runs_comparison.csv, .md and .png, computing validation ROC-AUC from the saved pipelines. The supplied historical chart includes a reproduction; a fresh standard batch chart has two candidates. These scripts require a successful local training run first; supplied historical report IDs alone do not recreate a database. The added wrapper scripts were syntax-checked; this separate three-command workflow has not been executed end-to-end.

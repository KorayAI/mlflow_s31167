# s31167 — validation comparison

F1 and ROC-AUC both use malignant (0) as the positive class. ROC-AUC is computed from saved model probabilities on each run's original validation split. These are measured results, not values copied from the reference image.

| Run | Validation F1 | Validation ROC-AUC |
| --- | ---: | ---: |
| baseline-logreg | 0.981132 | 0.997694 |
| rerun-of-7760b996 | 0.981132 | 0.997694 |
| rf-100-d5 | 0.933333 | 0.993291 |

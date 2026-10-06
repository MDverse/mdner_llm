import ast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def evaluate_confidence_separation(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Flatten TP and FP score lists into prediction records
    records = []
    for _, row in df.iterrows():
        # Parse serialized string representations of lists if necessary
        tp_raw = (
            ast.literal_eval(row["tp_scores"])
            if isinstance(row["tp_scores"], str)
            else row["tp_scores"]
        )
        fp_raw = (
            ast.literal_eval(row["fp_scores"])
            if isinstance(row["fp_scores"], str)
            else row["fp_scores"]
        )
        model_name = row.get("model", row.get("model_name"))
        # Append true positives (label = 1)
        if tp_raw is not None and len(tp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 1,
                    }
                    for score in tp_raw
                ]
            )
        # Append false positives (label = 0)
        if fp_raw is not None and len(fp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 0,
                    }
                    for score in fp_raw
                ]
            )
    preds_df = pd.DataFrame(records)
    # 2. Compute correlation and separation metrics per model
    table_rows = []
    for model_name, group in preds_df.groupby("model"):
        y_true = group["is_tp"].to_numpy()
        conf_scores = group["confidence"].to_numpy()
        # Point-biserial measures linear relationship between continuous score and binary label
        r_pb, p_pb = stats.pointbiserialr(y_true, conf_scores)
        # ROC-AUC measures probability that a random TP has a higher score than a random FP
        auc = roc_auc_score(y_true, conf_scores)
        table_rows.append(
            {
                "Model": model_name,
                "Number of True Positives": int(np.sum(y_true == 1)),
                "Number of False Positives": int(np.sum(y_true == 0)),
                "Point-Biserial Correlation": r_pb,
                "P-Value": f"{p_pb:.1e}",
                "ROC-AUC": auc,
            }
        )
    return (
        pd.DataFrame(table_rows)
        .sort_values(by="ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )

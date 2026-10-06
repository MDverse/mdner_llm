from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_aligned_data(
    llm_parquet_path: Path | str,
    gliner_parquet_path: Path | str,
    gliner_model_name: str,
    best_llm_name: str,
) -> pd.DataFrame:
    """Charge et aligne les données des deux modèles par texte et catégorie."""
    df_llm = pd.read_parquet(llm_parquet_path)
    df_gliner = pd.read_parquet(gliner_parquet_path)

    # Filtrage
    df_llm = df_llm[df_llm["model_name"] == best_llm_name].copy()
    mask_gliner = df_gliner["model_name"].str.contains(gliner_model_name, regex=False)
    df_gliner = df_gliner[mask_gliner].copy()
    df_llm["text_id"] = df_llm["text"].apply(lambda x: Path(x).name)
    df_gliner["text_id"] = df_gliner["response_metadata"].apply(lambda x: Path(x).name)

    text_id_col = (
        "response_metadata" if "response_metadata" in df_llm.columns else "text"
    )

    # Renommage des colonnes clés pour la fusion
    cols_to_keep = [
        text_id_col,
        "category",
        "groundtruth_by_category",
        "prediction_by_category",
        "tp_entities",
        "fp_entities",
        "fn_entities",
    ]

    llm_sub = df_llm[cols_to_keep].rename(
        columns={
            "prediction_by_category": "pred_llm",
            "tp_entities": "tp_llm",
            "fp_entities": "fp_llm",
            "fn_entities": "fn_llm",
        }
    )

    gliner_sub = df_gliner[cols_to_keep].rename(
        columns={
            "prediction_by_category": "pred_gliner",
            "tp_entities": "tp_gliner",
            "fp_entities": "fp_gliner",
            "fn_entities": "fn_gliner",
        }
    )

    # Fusion externe pour conserver toutes les catégories vues par l'un ou l'autre
    merged = pd.merge(
        gliner_sub,
        llm_sub.drop(columns=["groundtruth_by_category"]),
        on=[text_id_col, "category"],
        how="outer",
    )

    # Remplacement des valeurs NaN par des ensembles vides
    for col in [
        "groundtruth_by_category",
        "pred_gliner",
        "pred_llm",
        "tp_gliner",
        "tp_llm",
        "fp_gliner",
        "fp_llm",
        "fn_gliner",
        "fn_llm",
    ]:
        merged[col] = merged[col].apply(
            lambda x: set(x) if isinstance(x, (set, list, np.ndarray)) else set()
        )

    merged["text_id"] = merged[text_id_col]
    return merged


# =====================================================================
# 1. Niveau Texte : Concordance sur le nombre et la présence des classes
# =====================================================================
def analyze_text_level_class_agreement(
    df_aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Évalue si les deux modèles détectent le même nombre de classes uniques par texte."""
    records = []

    for text_id, group in df_aligned.groupby("text_id"):
        # Classes où chaque modèle a prédit au moins une entité
        cats_gliner = set(group[group["pred_gliner"].apply(len) > 0]["category"])
        cats_llm = set(group[group["pred_llm"].apply(len) > 0]["category"])
        cats_gt = set(
            group[group["groundtruth_by_category"].apply(len) > 0]["category"]
        )

        records.append(
            {
                "text_id": text_id,
                "nb_classes_gt": len(cats_gt),
                "nb_classes_gliner": len(cats_gliner),
                "nb_classes_llm": len(cats_llm),
                "classes_count_diff": len(cats_gliner) - len(cats_llm),
                "same_class_count": len(cats_gliner) == len(cats_llm),
                "exact_class_set_match": cats_gliner == cats_llm,
                "jaccard_classes": (
                    len(cats_gliner & cats_llm) / len(cats_gliner | cats_llm)
                    if (cats_gliner | cats_llm)
                    else 1.0
                ),
            }
        )

    df_text_agreement = pd.DataFrame(records)
    pct_same_count = df_text_agreement["same_class_count"].mean() * 100
    pct_exact_match = df_text_agreement["exact_class_set_match"].mean() * 100

    print("=== Concordance globale au niveau Texte ===")
    print(f"Même nombre de classes détectées : {pct_same_count:.1f}% des textes")
    print(f"Ensemble exact de classes identique : {pct_exact_match:.1f}% des textes")
    print(
        f"Similarité moyenne de Jaccard des classes : {df_text_agreement['jaccard_classes'].mean():.3f}"
    )

    return df_text_agreement, pct_exact_match


# =====================================================================
# 2. Niveau Catégorie & Entité : Recouvrement et complémentarité
# =====================================================================
def compute_detailed_category_overlap(
    df_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule pour chaque catégorie :

    - Les entités trouvées en commun
    - Les entités exclusives à chaque modèle
    - La part de vérité terrain capturée exclusivement par l'un ou l'autre
    """
    cat_summary = []

    for category, group in df_aligned.groupby("category"):
        total_gt = sum(len(s) for s in group["groundtruth_by_category"])

        # Cumul des entités prédites
        all_pred_gliner = set().union(*group["pred_gliner"])
        all_pred_llm = set().union(*group["pred_llm"])

        # Vrais positifs (TP)
        tp_gliner_total = set().union(*group["tp_gliner"])
        tp_llm_total = set().union(*group["tp_llm"])

        # Recouvrement TP (Qui a trouvé quoi de correct ?)
        shared_tp = tp_gliner_total & tp_llm_total
        gliner_only_tp = tp_gliner_total - tp_llm_total
        llm_only_tp = tp_llm_total - tp_gliner_total
        missed_by_both = (
            total_gt - len(shared_tp) - len(gliner_only_tp) - len(llm_only_tp)
        )

        # Faux positifs (FP / Bruit / Hallucinations)
        fp_gliner_total = set().union(*group["fp_gliner"])
        fp_llm_total = set().union(*group["fp_llm"])
        shared_fp = fp_gliner_total & fp_llm_total
        gliner_only_fp = fp_gliner_total - fp_llm_total
        llm_only_fp = fp_llm_total - fp_gliner_total

        cat_summary.append(
            {
                "category": category,
                "total_groundtruth": total_gt,
                "both_correct (TP ∩ TP)": len(shared_tp),
                "gliner_unique_correct": len(gliner_only_tp),
                "llm_unique_correct": len(llm_only_tp),
                "missed_by_both (FN)": max(0, missed_by_both),
                "shared_hallucinations (FP)": len(shared_fp),
                "gliner_only_fp": len(gliner_only_fp),
                "llm_only_fp": len(llm_only_fp),
                "gliner_recall": (len(tp_gliner_total) / total_gt if total_gt else 0),
                "llm_recall": len(tp_llm_total) / total_gt if total_gt else 0,
                "ensemble_oracle_recall": (
                    len(tp_gliner_total | tp_llm_total) / total_gt if total_gt else 0
                ),
            }
        )

    df_cat = pd.DataFrame(cat_summary).sort_values(
        by="total_groundtruth", ascending=False
    )
    return df_cat


# =====================================================================
# 3. Visualisation : Recouvrement des TP et gains complémentaires
# =====================================================================
def plot_complementarity_profile(
    df_cat_summary: pd.DataFrame,
    gliner_label: str,
    llm_label: str,
    output_fig_path: Path | str | None = None,
):
    """Génère un graphique montrant le partage des découvertes correctes (TP)

    et le potentiel de complémentarité par classe.
    """
    df = df_cat_summary.copy().set_index("category")

    # Calcul des pourcentages par rapport au ground truth total
    plot_data = pd.DataFrame(index=df.index)
    plot_data["Both Correct"] = (
        df["both_correct (TP ∩ TP)"] / df["total_groundtruth"]
    ) * 100
    plot_data[f"Only {gliner_label} Correct"] = (
        df["gliner_unique_correct"] / df["total_groundtruth"]
    ) * 100
    plot_data[f"Only {llm_label} Correct"] = (
        df["llm_unique_correct"] / df["total_groundtruth"]
    ) * 100
    plot_data["Missed by Both"] = (
        df["missed_by_both (FN)"] / df["total_groundtruth"]
    ) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2CA02C", "#1F77B4", "#FF7F0E", "#D62728"]
    bottom = pd.Series(0.0, index=plot_data.index)

    for col, color in zip(plot_data.columns, colors):
        ax.barh(
            plot_data.index,
            plot_data[col],
            left=bottom,
            color=color,
            label=col,
            height=0.6,
        )
        bottom += plot_data[col]

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of Ground Truth Entities", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Légende en haut
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    if output_fig_path:
        Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_fig_path, bbox_inches="tight", dpi=300)
    plt.show()

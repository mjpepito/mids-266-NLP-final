"""
Evaluation metric utilities for the Gen Z NLP project.
"""

from typing import Dict

import numpy as np
import pandas as pd
import evaluate
from bert_score import score as bertscore_fn


def evaluate_translations(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Compute ROUGE and BERTScore metrics for a prediction dataframe."""
    rouge = evaluate.load("rouge")
    predictions = pred_df["prediction"].fillna("").astype(str).tolist()
    references = pred_df["reference"].fillna("").astype(str).tolist()

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    P, R, F1 = bertscore_fn(predictions, references, lang="en", verbose=False)

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore_f1": float(F1.mean().item()),
    }


def score_candidate_consistency(
    source_profile: Dict,
    candidate_profile: Dict,
) -> Dict:
    """Score how well a translation candidate preserves the source behavioral profile."""
    act_match = 1.0 if source_profile["act_id"] == candidate_profile["act_id"] else 0.0
    emo_match = 1.0 if source_profile["emo_id"] == candidate_profile["emo_id"] else 0.0

    act_sim = float(np.dot(source_profile["act_probs"], candidate_profile["act_probs"]))
    emo_sim = float(np.dot(source_profile["emo_probs"], candidate_profile["emo_probs"]))

    return {
        "act_match": act_match,
        "emo_match": emo_match,
        "act_soft_sim": act_sim,
        "emo_soft_sim": emo_sim,
        "consistency_score": (
            0.4 * act_sim + 0.4 * emo_sim
            + 0.1 * act_match + 0.1 * emo_match
        ),
        "src_act": source_profile["act_label"],
        "src_emo": source_profile["emo_label"],
        "cand_act": candidate_profile["act_label"],
        "cand_emo": candidate_profile["emo_label"],
    }

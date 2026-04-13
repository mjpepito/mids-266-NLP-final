"""
Model training and inference helpers for the Gen Z NLP project.
"""

from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Label maps
ACT_LABELS = {
    0: "dummy", 1: "inform", 2: "question",
    3: "directive", 4: "commissive",
}
EMO_LABELS = {
    0: "no_emotion", 1: "anger", 2: "disgust",
    3: "fear", 4: "happiness", 5: "sadness", 6: "surprise",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_act_emotion(
    text: str,
    act_model: AutoModelForSequenceClassification,
    emo_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
) -> Dict:
    """Return predicted act and emotion labels for a single utterance."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    ).to(DEVICE)

    with torch.no_grad():
        act_logits = act_model(**enc).logits
        emo_logits = emo_model(**enc).logits

    act_id = int(act_logits.argmax(dim=-1).item())
    emo_id = int(emo_logits.argmax(dim=-1).item())
    return {
        "act_id": act_id,
        "act_label": ACT_LABELS[act_id],
        "emo_id": emo_id,
        "emo_label": EMO_LABELS[emo_id],
        "act_probs": torch.softmax(act_logits, dim=-1).cpu().numpy().tolist()[0],
        "emo_probs": torch.softmax(emo_logits, dim=-1).cpu().numpy().tolist()[0],
    }

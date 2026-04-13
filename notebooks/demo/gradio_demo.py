#!/usr/bin/env python3
"""
Gen Z ↔ Standard English Translator — Gradio Demo

Standalone demo script that loads models from HuggingFace Hub
(or local directory) and launches the interactive Gradio UI.

Usage:
    python gradio_demo.py                          # load from HuggingFace Hub
    python gradio_demo.py --local --model-dir ../../models  # load from local
    python gradio_demo.py --share                  # generate public link
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ── Config ────────────────────────────────────────────────────────
HF_USERNAME = "mjpepito"
HF_REPOS = {
    "bertweet_act_emo": f"{HF_USERNAME}/genz-bertweet-act-emo",
    "bart_translation": f"{HF_USERNAME}/genz-bart-translation",
}

ACT_LABELS = {
    0: "dummy", 1: "inform", 2: "question",
    3: "directive", 4: "commissive",
}
EMO_LABELS = {
    0: "no_emotion", 1: "anger", 2: "disgust",
    3: "fear", 4: "happiness", 5: "sadness", 6: "surprise",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load models ───────────────────────────────────────────────────
def load_models_hf():
    """Load models from HuggingFace Hub."""
    print("Loading models from HuggingFace Hub...")

    bart_tokenizer = AutoTokenizer.from_pretrained(
        HF_REPOS["bart_translation"], subfolder="tokenizer"
    )
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(
        HF_REPOS["bart_translation"], subfolder="model"
    ).to(DEVICE).eval()

    bertweet_tokenizer = AutoTokenizer.from_pretrained(
        HF_REPOS["bertweet_act_emo"], subfolder="tokenizer",
        use_fast=False,
    )
    act_model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPOS["bertweet_act_emo"], subfolder="act_model"
    ).to(DEVICE).eval()
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPOS["bertweet_act_emo"], subfolder="emo_model"
    ).to(DEVICE).eval()

    print("✅ All models loaded from HuggingFace Hub")
    return bart_tokenizer, bart_model, bertweet_tokenizer, act_model, emo_model


def load_models_local(model_dir: Path):
    """Load models from local directory."""
    print(f"Loading models from {model_dir.resolve()}...")

    bart_tokenizer = AutoTokenizer.from_pretrained(model_dir / "bart_tokenizer")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir / "bart_translation_model"
    ).to(DEVICE).eval()

    bertweet_tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "bertweet_tokenizer", use_fast=False
    )
    act_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir / "act_bertweet_model"
    ).to(DEVICE).eval()
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir / "emo_bertweet_model"
    ).to(DEVICE).eval()

    print("✅ All models loaded from local disk")
    return bart_tokenizer, bart_model, bertweet_tokenizer, act_model, emo_model


# ── Inference helpers ─────────────────────────────────────────────
def predict_act_emotion(text, bertweet_tokenizer, act_model, emo_model):
    enc = bertweet_tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=128, padding="max_length",
    ).to(DEVICE)
    with torch.no_grad():
        act_logits = act_model(**enc).logits
        emo_logits = emo_model(**enc).logits
    act_id = int(act_logits.argmax(dim=-1).item())
    emo_id = int(emo_logits.argmax(dim=-1).item())
    return {
        "act_id": act_id, "act_label": ACT_LABELS[act_id],
        "emo_id": emo_id, "emo_label": EMO_LABELS[emo_id],
        "act_probs": torch.softmax(act_logits, dim=-1).cpu().numpy().tolist()[0],
        "emo_probs": torch.softmax(emo_logits, dim=-1).cpu().numpy().tolist()[0],
    }


def generate_candidates(text, direction, bart_tokenizer, bart_model, n=8):
    prefix = (
        "translate to Gen Z: "
        if direction == "standard_to_genz"
        else "translate to standard English: "
    )
    inputs = bart_tokenizer(
        prefix + text, return_tensors="pt", max_length=128, truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = bart_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=max(n, 8),
            num_return_sequences=n,
            diversity_penalty=0.8,
            num_beam_groups=min(n, 4),
            no_repeat_ngram_size=3,
        )
    candidates = [bart_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return list(dict.fromkeys(candidates))


def score_consistency(src_profile, cand_profile):
    act_match = 1.0 if src_profile["act_id"] == cand_profile["act_id"] else 0.0
    emo_match = 1.0 if src_profile["emo_id"] == cand_profile["emo_id"] else 0.0
    act_sim = float(np.dot(src_profile["act_probs"], cand_profile["act_probs"]))
    emo_sim = float(np.dot(src_profile["emo_probs"], cand_profile["emo_probs"]))
    return {
        "act_match": act_match, "emo_match": emo_match,
        "consistency_score": 0.4 * act_sim + 0.4 * emo_sim + 0.1 * act_match + 0.1 * emo_match,
        "cand_act": cand_profile["act_label"],
        "cand_emo": cand_profile["emo_label"],
    }


# ── Gradio pipeline ──────────────────────────────────────────────
def build_demo(bart_tok, bart_mdl, bt_tok, act_mdl, emo_mdl):

    def pipeline(text, direction):
        if not text.strip():
            return "Please enter some text.", "", pd.DataFrame()

        dir_key = (
            "genz_to_standard"
            if direction == "Gen Z → Standard English"
            else "standard_to_genz"
        )

        src = predict_act_emotion(text, bt_tok, act_mdl, emo_mdl)
        candidates = generate_candidates(text, dir_key, bart_tok, bart_mdl)

        rows = []
        for rank, cand in enumerate(candidates, 1):
            cp = predict_act_emotion(cand, bt_tok, act_mdl, emo_mdl)
            sc = score_consistency(src, cp)
            rows.append({
                "Rank": f"{'🏆 ' if rank == 1 else ''}#{rank}",
                "Translation": cand,
                "Act": sc["cand_act"],
                "Emotion": sc["cand_emo"],
                "Score": f'{sc["consistency_score"]:.3f}',
                "Act ✓": "✅" if sc["act_match"] == 1.0 else "❌",
                "Emo ✓": "✅" if sc["emo_match"] == 1.0 else "❌",
            })

        cand_df = pd.DataFrame(rows).sort_values(
            "Score", ascending=False
        ).reset_index(drop=True)
        best = cand_df.iloc[0]

        act_probs = src["act_probs"]
        emo_probs = src["emo_probs"]
        source_md = (
            f"**Source text:** {text}\n\n"
            f"**Act:** {src['act_label']}\n\n"
            f"**Emotion:** {src['emo_label']}\n\n"
            f"**Act probs:** dummy={act_probs[0]:.3f}, inform={act_probs[1]:.3f}, "
            f"question={act_probs[2]:.3f}, directive={act_probs[3]:.3f}, "
            f"commissive={act_probs[4]:.3f}\n\n"
            f"**Emo probs:** none={emo_probs[0]:.3f}, anger={emo_probs[1]:.3f}, "
            f"disgust={emo_probs[2]:.3f}, fear={emo_probs[3]:.3f}, "
            f"happiness={emo_probs[4]:.3f}, sadness={emo_probs[5]:.3f}, "
            f"surprise={emo_probs[6]:.3f}"
        )

        best_md = (
            f"### 🏆 Best Translation\n\n"
            f"**{best['Translation']}**\n\n---\n\n"
            f"**Act:** {best['Act']} | **Emotion:** {best['Emotion']}\n\n"
            f"**Consistency score:** {best['Score']}\n\n"
            f"**Act match:** {best['Act ✓']} | **Emo match:** {best['Emo ✓']}"
        )

        return best_md, source_md, cand_df

    with gr.Blocks(
        title="Gen Z ↔ English Translator + Action/Emotion Analyzer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🗣️ Gen Z ↔ Standard English Translator\n\n"
            "Translate between Gen Z slang and standard English, with "
            "**action/emotion consistency** analysis.\n\n"
            "The system generates multiple candidates, scores each against "
            "the source behavioral profile, and selects the best match."
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Enter text",
                    placeholder="e.g., No cap, this guy is highkey excited today.",
                    lines=3,
                )
                direction = gr.Radio(
                    choices=["Gen Z → Standard English", "Standard English → Gen Z"],
                    value="Gen Z → Standard English",
                    label="Translation Direction",
                )
                btn = gr.Button("🔄 Translate & Analyze", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                best_output = gr.Markdown(label="Best Translation")
            with gr.Column(scale=1):
                source_profile = gr.Markdown(label="Source Profile")

        gr.Markdown("### 📊 All Candidates (ranked by behavioral consistency)")
        candidate_table = gr.Dataframe(label="Candidates", interactive=False)

        btn.click(
            fn=pipeline,
            inputs=[input_text, direction],
            outputs=[best_output, source_profile, candidate_table],
        )

        gr.Examples(
            examples=[
                ["No cap, this guy is highkey excited today.", "Gen Z → Standard English"],
                ["That outfit is fire no cap", "Gen Z → Standard English"],
                ["She ate and left no crumbs", "Gen Z → Standard English"],
                ["I'm honestly really tired and a bit annoyed.", "Standard English → Gen Z"],
                ["She did an excellent job on the presentation.", "Standard English → Gen Z"],
                ["That was very embarrassing and awkward.", "Standard English → Gen Z"],
            ],
            inputs=[input_text, direction],
        )

    return demo


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen Z Translation Demo")
    parser.add_argument(
        "--local", action="store_true",
        help="Load models from local directory instead of HuggingFace Hub",
    )
    parser.add_argument(
        "--model-dir", type=str, default="../../models",
        help="Path to local model directory (only used with --local)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Generate a public Gradio link",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Local port for the Gradio server",
    )
    args = parser.parse_args()

    if args.local:
        models = load_models_local(Path(args.model_dir))
    else:
        models = load_models_hf()

    bart_tok, bart_mdl, bt_tok, act_mdl, emo_mdl = models
    demo = build_demo(bart_tok, bart_mdl, bt_tok, act_mdl, emo_mdl)
    demo.launch(share=args.share, server_port=args.port)

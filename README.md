# Gen Z Slang Translation + Action/Emotion Analysis

**UC Berkeley MIDS — W266 Natural Language Processing Final Project**

## Overview

This project builds a unified NLP pipeline that:
1. **Classifies dialogue acts and emotions** at the utterance level (DailyDialog)
2. **Translates bidirectionally** between Gen Z slang and standard English (T5 baseline → BART)
3. **Reranks translation candidates** by behavioral consistency — selecting translations that preserve the source's communicative act and emotional profile

### Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| BART translation (Gen Z → Standard) | ROUGE-1 | 0.806 |
| BART translation (Gen Z → Standard) | BERTScore F1 | 0.976 |
| BART translation (Standard → Gen Z) | ROUGE-1 | 0.680 |
| BART translation (Standard → Gen Z) | BERTScore F1 | 0.948 |
| Act classification (BERTweet) | Accuracy | 0.887 |
| Emotion classification (BERT) | Macro-F1 | 0.430 |

> **Note:** Emotion classification is severely affected by class imbalance in DailyDialog (~83% "no emotion"). See the paper for a detailed analysis and proposed remedies.

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/mjpepito/mids-266-NLP-final.git
cd mids-266-NLP-final
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `notebooks/GenZ_DailyDialog_Unified.ipynb` in **Google Colab** (recommended for GPU) or Jupyter.

- **Colab:** Upload data files from `data/raw/` when prompted, or mount Google Drive.
- **Local:** Paths resolve automatically when running from the `notebooks/` directory.

### 4. Launch the demo

**Option A — From the notebook:**
Run the Gradio cell at the end of the notebook. It generates a public shareable link.

**Option B — Standalone script** (after training):
```bash
cd notebooks/demo
python gradio_demo.py                           # loads models from ../../models/
python gradio_demo.py --share                   # with public link
python gradio_demo.py --model-dir /path/to/models --port 8080
```

## Project Structure

```
mids-266-NLP-final/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── notebooks/
│   ├── GenZ_DailyDialog_Unified.ipynb  # Main unified notebook (final)
│   ├── demo/
│   │   └── gradio_demo.py             # Standalone Gradio demo script
│   └── exploratory/                    # Earlier / individual notebooks
│       ├── 266_Final_MP.ipynb
│       ├── DailyDialog_baseline.ipynb
│       ├── DailyDialog_BERTweet.ipynb
│       ├── GenZ_BART_baseline.ipynb
│       └── GenZ_BART_bidirectional.ipynb
│
├── data/
│   ├── raw/                            # Source CSV files
│   │   ├── dailydialog_train.csv
│   │   ├── dailydialog_validation.csv
│   │   ├── dailydialog_test.csv
│   │   ├── GenZ_Translations.csv
│   │   └── GenZ_500_slang_terms.csv
│   └── processed/                      # Generated during runs (gitignored)
│
├── models/                             # Saved checkpoints (gitignored)
│
├── paper/
│   ├── GenZ_NLP_Research_Paper.md      # Research paper (markdown)
│   └── figures/                        # Plots and figures
│
├── references/                         # Course guidelines & reference papers
│   ├── 266_Final_Project_Guidelines.pdf
│   ├── 266_How_to_Write_Project_Paper.pdf
│   └── Gen_Alpha_Slang_ACL2025.pdf
│
└── src/                                # Reusable Python utilities
    ├── __init__.py
    ├── data_utils.py                   # Data loading & preprocessing
    ├── models.py                       # Model inference helpers
    └── evaluation.py                   # Evaluation metric functions
```

## Data Sources

| Source | Type | Size |
|--------|------|------|
| [DailyDialog](http://yanran.li/dailydialog) | Multi-turn dialogue with act/emotion labels | 13,118 dialogues |
| [MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) | Slang lexicon | HuggingFace |
| [Programmer-RD-AI/genz-slang-pairs-1k](https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k) | Paired sentence rewrites | 1,000 pairs |
| [thesherrycode/gen-z-slangs-translation](https://huggingface.co/datasets/thesherrycode/gen-z-slangs-translation) | Short paired translations | HuggingFace |
| GenZ_Translations.csv | Curated parallel pairs | local |
| GenZ_500_slang_terms.csv | Slang terms + definitions | 500 terms |

## Models

| Model | Task | Checkpoint |
|-------|------|------------|
| BERT | Act/Emotion classification (baseline) | `bert-base-uncased` |
| BERTweet | Act/Emotion classification (informal text) | `vinai/bertweet-base` |
| T5 | Translation baseline | `t5-small` |
| BART | Translation (improved) | `facebook/bart-base` |

## Pre-trained Models

All trained model weights are hosted publicly on HuggingFace Hub:

| Repo | Contents |
|------|----------|
| [`mjpepito/genz-bert-act-emo`](https://huggingface.co/mjpepito/genz-bert-act-emo) | BERT tokenizer + act classifier + emotion classifier |
| [`mjpepito/genz-bertweet-act-emo`](https://huggingface.co/mjpepito/genz-bertweet-act-emo) | BERTweet tokenizer + act classifier + emotion classifier |
| [`mjpepito/genz-bart-translation`](https://huggingface.co/mjpepito/genz-bart-translation) | BART tokenizer + bidirectional translation model |

The notebook and demo script load these automatically — no manual download needed.

## Interactive Demo

The Gradio demo lets you:
- Enter text in Gen Z slang or standard English
- Choose a translation direction
- See ranked translation candidates with act/emotion consistency scores
- Inspect behavioral profiles of source and translated text

Run it from the notebook or use the standalone script at `notebooks/demo/gradio_demo.py`.

## Authors

- Marvin Pepito
- Jeevan Maddila
- Ethan Chen

## License

This project was developed for academic purposes as part of UC Berkeley's Master of Information and Data Science (MIDS) program.

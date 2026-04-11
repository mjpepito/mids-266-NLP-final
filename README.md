# Gen Z Slang Translation + Action/Emotion Analysis

**UC Berkeley MIDS — W266 Natural Language Processing Final Project**

## Overview

This project builds a unified NLP pipeline that:
1. **Classifies dialogue acts and emotions** at the utterance level (DailyDialog)
2. **Translates bidirectionally** between Gen Z slang and standard English (T5 baseline → BART)
3. **Reranks translation candidates** by behavioral consistency — selecting translations that preserve the source's communicative act and emotional profile

### Key Results
- BART translation: ROUGE-1 0.81, BERTScore F1 0.98 (Gen Z → Standard)
- Behavioral reranking pipeline connects translation with pragmatic analysis
- Detailed analysis of emotion classification challenges due to class imbalance

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
Open `notebooks/GenZ_DailyDialog_Unified.ipynb` in Google Colab or Jupyter.

Upload the data files from `data/raw/` when prompted (Colab), or adjust paths for local execution.

### 4. Launch the demo
Run the Gradio cell at the end of the notebook, or:
```bash
python notebooks/demo/gradio_demo.py
```

## Project Structure

```
notebooks/          — Main notebook + exploratory notebooks
data/raw/           — Source CSV files
data/processed/     — Generated during notebook runs (gitignored)
models/             — Saved model checkpoints (gitignored)
paper/              — Research paper (markdown + PDF)
references/         — Course guidelines and reference papers
src/                — Optional reusable Python modules
```

## Data Sources

| Source | Type | Size |
|--------|------|------|
| DailyDialog | Multi-turn dialogue with act/emotion labels | 13,118 dialogues |
| MLBtrio/genz-slang-dataset | Slang lexicon (HuggingFace) | loaded via `datasets` |
| Programmer-RD-AI/genz-slang-pairs-1k | Paired sentence rewrites (HF) | 1,000 pairs |
| thesherrycode/gen-z-slangs-translation | Short paired translations (HF) | small |
| GenZ_Translations.csv | Curated parallel pairs | local |
| GenZ_500_slang_terms.csv | Slang terms + definitions | 500 terms |

## Models

| Model | Task | Checkpoint |
|-------|------|------------|
| BERT | Act/Emotion classification baseline | bert-base-uncased |
| BERTweet | Act/Emotion classification (informal text) | vinai/bertweet-base |
| T5 | Translation baseline | t5-small |
| BART | Translation (improved) | facebook/bart-base |

## Interactive Demo

The Gradio demo lets you:
- Enter text in Gen Z slang or standard English
- Choose translation direction
- See ranked translation candidates with act/emotion consistency scores
- Inspect behavioral profiles of source and translated text

## Authors

- Ethan Chen
- Jeevan Maddila
- Marvin Pepito

## License

This project was developed for academic purposes as part of UC Berkeley's MIDS program.
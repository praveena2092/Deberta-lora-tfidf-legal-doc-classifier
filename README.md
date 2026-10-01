# DeBERTa-LoRA + TF-IDF Blend for Legal & Policy Document Classification

A hybrid classifier for 30-class legal and policy document categorization: a LoRA-fine-tuned DeBERTa encoder blended with a calibrated TF-IDF + LinearSVM model.

## Task

Given a document's text, predict one of 30 categories. Metric: **accuracy**.

| | |
|---|---|
| Train documents | 50,840 (labeled) |
| Test documents | 12,710 |
| Classes | 30 (imbalanced) |

## Approach

1. **Cleaning + pre-slice** – whitespace/page-marker cleanup, then keep the first 400 and last 1,200 words.
2. **Transformer branch** – a DeBERTa model fine-tuned with **LoRA** (parameter-efficient) and merged into the base weights. Inputs use **head + tail truncation** (first 128 + last 384 tokens, 512 total). The notebook loads the merged model and produces class probabilities for validation and test sets.
3. **Lexical branch** – `TfidfVectorizer` (50k features, 1–2-grams) → `LinearSVC(class_weight="balanced")` wrapped in `CalibratedClassifierCV` for probabilities.
4. **Blend** – weighted average of the two probability matrices; the weight is grid-searched (0 → 1, step 0.05) on the validation set.

## Results (validation, 7,626 docs)

| Model | Accuracy |
|---|---|
| DeBERTa-LoRA (solo) | 0.6663 |
| TF-IDF + calibrated LinearSVM (solo) | 0.7257 |
| **Blend** (0.25 × DeBERTa + 0.75 × TF-IDF) | **0.7278** |

Takeaway: on these long documents a strong lexical baseline beats a 512-token transformer view, and blending adds a small further gain. The blend weight was tuned on the same validation set it is reported on, so the 0.7278 is slightly optimistic.

## Repository layout

```
notebooks/deberta_lora_tfidf_blend.ipynb   # inference, TF-IDF model, blending, submission
data/                                      # put train.csv / test.csv here (git-ignored)
requirements.txt
```

## Important: LoRA training step

This notebook covers **inference, the TF-IDF model, and blending**. It loads an already-trained, merged LoRA model from `YOUR_OUTPUT_PATH` (the LoRA training run itself was done in a separate notebook and is not included here). To reproduce end-to-end, first fine-tune DeBERTa with LoRA using the same preprocessing/splits (seed 42, 15% stratified validation), merge the adapter (`merge_and_unload()`), save it, and point `YOUR_OUTPUT_PATH` at it.

The notebook also has a cell comparing the new solo predictions with a previous `submission.csv` (99.96% agreement in the original run) — remove or skip it if you don't have a prior submission.

## Usage

```bash
pip install -r requirements.txt
```

Open `notebooks/deberta_lora_tfidf_blend.ipynb`, set `train_path`, `test_path`, and `YOUR_OUTPUT_PATH`, and run all cells. Output: `submission.csv` (`ID`, `label`).

## License

MIT

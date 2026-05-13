# An Encoder-Decoder Transformer for Morphological Inflection

CL2 Coursework | Task 1 | English (SIGMORPHON 2023)

## Model

An encoder-decoder transformer trained on English data from the SIGMORPHON 2023 morphological inflection shared task. Our final model uses $d_{\text{model}}=64$ (250k parameters), selected via ablation over three model dimensions (see below).

The encoder reads the concatenation of a lemma and morphological feature string; the decoder generates the inflected form autoregressively.

- **Input**: `lemma + SEP + features` (byte-level, +4 shifted)
- **Output**: inflected form (byte-level, autoregressive decoding)
- **Vocabulary size**: 260 (256 bytes + 4 special tokens: PAD=0, BOS=1, EOS=2, SEP=3)

## Requirements

Python 3.9+ and PyTorch 2.0+.

```bash
pip install -r requirements.txt
```

Data is fetched automatically from the SIGMORPHON 2023 repository on first run.

## Training

```bash
python train.py 64
```

Saves the best checkpoint (by validation loss) to `best_model_64.pth`.

## Evaluation

```bash
python evaluate.py 64
```

Uses beam search decoding (beam size 4). Reports exact match accuracy and average Levenshtein distance. Predictions are saved to `predictions_64.tsv`.

## Ablation Analysis

To reproduce the model selection experiment, first train all three variants (each saves its own checkpoint), then evaluate:

```bash
python train.py 128 && python train.py 64 && python train.py 32
```

```bash
python evaluate.py 128 && python evaluate.py 64 && python evaluate.py 32
```

All three sizes share the same fixed hyperparameters (2 encoder/decoder layers, dropout=0.3, LR=1e-3, patience=20). Results:

| $d_{\text{model}}$ | Params | Best epoch | Exact match | Avg ED |
|---|---|---|---|---|
| 128 | 828k | 8 | 0.902 | 0.181 |
| **64** | **250k** | **19** | **0.907** | **0.176** |
| 32 | 84k | 29 | 0.893 | 0.216 |

## File Structure

```
├── data.py               tokenisation, Dataset, DataLoader
├── model.py              all nn.Module classes
├── train.py              training entry point (accepts size argument)
├── evaluate.py           evaluation entry point (accepts size argument)
├── requirements.txt
└── README.md
```

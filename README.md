# Sarcasm Detection (NLP Project)

This repository explores several ML/DL approaches for sarcasm detection on the **Sarcasm Headlines Dataset** (Onion/HuffPost headlines). We implemented and compared models ranging from classical baselines to transformer-based models.

## Notebooks
- `LogReg+SVM.ipynb`
- `LSTM.ipynb` (my contribution)
- `BiLSTM.ipynb`
- `BERT_baseline.ipynb`
- `BERT_improved.ipynb`
- `GUI_demo+words_importance.ipynb`

## My Contribution (LSTM)
I implemented and evaluated the LSTM sarcasm detector, including:
- Data preprocessing + vocabulary construction
- Baseline LSTM (random embeddings) vs improved LSTM (GloVe-100d init, fine-tuned)
- Training loop + validation + early stopping
- Metrics + confusion matrices
- Error analysis (false positives/negatives)
- Word-importance visualization (leave-one-out masking)

## Dataset
- `Sarcasm_Headlines_Dataset.json` (included)

## Report
- `Final Report_Sarcasm Detection.pdf`

## Running
Open any notebook and run cells top-to-bottom.

Notes for `LSTM.ipynb`:
- Uses PyTorch, NumPy/Pandas, scikit-learn, matplotlib/seaborn
- Downloads **GloVe 6B 100d** on first run (large download) and reuses it on later runs
- If you want a clean setup, create a Python venv and install dependencies from `requirements.txt`

### Install
Create a venv, then:

```bash
pip install -r requirements.txt
```

PyTorch install depends on your machine (CPU vs GPU). Install it from the official selector:
https://pytorch.org/get-started/locally/


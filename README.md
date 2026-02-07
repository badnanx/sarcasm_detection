# Sarcasm Detection (NLP Project)

This project explores multiple machine learning and deep learning approaches for sarcasm detection using the **Sarcasm Headlines Dataset**. Our team implemented and compared a range of models, from classical baselines to modern neural architectures.

## Models Included
- Logistic Regression + SVM
- LSTM (my contribution)
- BiLSTM
- BERT Baseline
- BERT Improved
- GUI demo with word‑importance visualization

## My Contribution
I was responsible for implementing and evaluating the **LSTM‑based sarcasm detector**, including:
- Data preprocessing and vocabulary construction  
- Random‑initialized and GloVe‑initialized LSTM variants  
- Training loops, validation logic, and metric reporting  
- Misclassification analysis and word‑importance visualization  

## Dataset
- `Sarcasm_Headlines_Dataset.json` (included)

## Project Report
- `Final Report_Sarcasm Detection.pdf` summarizes the full methodology and results.

## How to Run
Open any notebook in this repository and run all cells.  
The LSTM notebook requires:
- PyTorch  
- NumPy / Pandas  
- GloVe embeddings (100d)  

# IMDb Sentiment Analysis with GRU and LSTM

A deep-learning project comparing GRU and LSTM recurrent neural networks for binary sentiment classification of IMDb movie reviews.

I developed this academic project at Aristotle University of Thessaloniki as part of the MSc program in Digital Media - Computational Intelligence.

## Overview

The project uses the IMDb dataset of 50,000 movie reviews, labeled as positive or negative. Reviews are tokenized with the `bert-base-uncased` tokenizer and classified using recurrent neural networks implemented in PyTorch.

## Models

Both models use:

- BERT token IDs with a maximum sequence length of 256
- A learned 128-dimensional embedding layer
- Two recurrent layers with 256 hidden units
- Dropout for regularization
- Binary cross-entropy with logits
- Adam optimization
- Early stopping based on validation loss
- A 70/15/15 train, validation, and test split

The experiment compares a **Gated Recurrent Unit (GRU)** with a **Long Short-Term Memory (LSTM)** network.

## Results

| Model | Test accuracy | Notes |
|---|---:|---|
| GRU | **88.69%** | Best overall result |
| LSTM | **86.81%** | Strong but slightly lower performance |

The GRU reached approximately 88% macro and weighted F1-score. Early stopping selected the best validation checkpoint and limited further overfitting.

## Evaluation

The notebook includes:

- Training and validation loss curves
- Test-set accuracy
- Confusion matrices
- Precision, recall, and F1-score
- Examples of correct and incorrect predictions
- GPU/CPU device selection

## Running the notebook

1. Download the IMDb Dataset CSV.
2. Update `dataset_path` inside the notebook to its local location.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open `imdb_sentiment_gru_lstm.ipynb` and run the cells in order.

The BERT tokenizer files are downloaded automatically on the first run.

## Repository files

- `imdb_sentiment_gru_lstm.ipynb` - GRU/LSTM training and evaluation
- `RNN143.pdf` - original Greek academic report
- `requirements.txt` - Python dependencies

## Limitations

The notebook currently uses local dataset paths that must be updated before execution. The BERT tokenizer is used for tokenization, while the recurrent models learn their own embeddings rather than using pretrained BERT representations. Generated model checkpoints are excluded from version control.

## Author

**Georgios Kolokotronis**  
MSc in Digital Media - Computational Intelligence  
Aristotle University of Thessaloniki

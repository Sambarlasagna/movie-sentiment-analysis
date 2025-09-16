# Movie Sentiment Analysis with CNN

This project implements a deep learning model using a Convolutional Neural Network (CNN) to analyze movie reviews and predict their sentiment (positive or negative). The model is trained and evaluated on the IMDB dataset using PyTorch and TorchText.

## Features

- Loads and preprocesses the IMDB dataset from HuggingFace Datasets.
- Tokenizes and numericalizes text data.
- Builds a vocabulary with special tokens and minimum frequency filtering.
- Uses pre-trained GloVe embeddings for word representation.
- Implements a CNN-based text classification model.
- Includes training, validation, and test splits with DataLoader support.
- Provides accuracy and loss tracking with visualization.
- Supports sentiment prediction for custom input text.

## Usage

1. **Clone the repository** and open `CNN_MSA.ipynb` in [Google Colab](https://colab.research.google.com/github/Sambarlasagna/movie-sentiment-analysis/blob/main/CNN_MSA.ipynb) or your local Jupyter environment.

2. **Install dependencies** (if running locally):
    ```sh
    pip install torch torchtext datasets matplotlib tqdm
    ```

3. **Run the notebook cells** step by step to:
    - Download and preprocess the IMDB dataset.
    - Build the vocabulary and DataLoaders.
    - Train and evaluate the CNN model.
    - Visualize training progress.
    - Predict sentiment for new movie reviews.

## Project Structure

- `CNN_MSA.ipynb` — Main Jupyter notebook containing all code, explanations, and experiments.
- `data/aclImdb_v1.tar.gz` — (Optional) Raw IMDB dataset archive (not used directly; data is loaded via HuggingFace Datasets).
- `.gitignore`, `debug.log` — Standard project files.

## Model Overview

The CNN model architecture includes:
- Embedding layer (with optional GloVe initialization)
- Multiple 1D convolutional layers with different filter sizes
- Max pooling and concatenation
- Fully connected output layer
- Dropout for regularization

## Example: Predicting Sentiment

After training, use the following code to predict sentiment for a custom review:

```python
text = "This film is fantastic! The story and acting were top-notch."
min_length = max(filter_sizes)
predict_sentiment(text, model, tokenizer, vocab, device, min_length, pad_index)
```

## References

- [PyTorch](https://pytorch.org/)
- [TorchText](https://pytorch.org/text/stable/index.html)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)

---

**Author:** [Your
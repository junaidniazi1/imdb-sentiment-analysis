# IMDB Sentiment Analysis with RNN/LSTM

Deep learning sentiment classifier for movie reviews using Recurrent Neural Networks (SimpleRNN, LSTM, GRU) on the IMDB dataset.

## 📋 Description

This project implements sentiment analysis on IMDB movie reviews using various recurrent neural network architectures. The model classifies reviews as positive or negative using TensorFlow/Keras and includes comprehensive evaluation with k-fold cross-validation.

## ✨ Features

- **Multiple RNN Architectures**: Support for SimpleRNN, LSTM, and GRU models
- **Bidirectional LSTM**: Enhanced context understanding with bidirectional processing
- **K-Fold Cross Validation**: Robust 5-fold cross-validation for reliable performance metrics
- **Regularization Techniques**: 
  - Dropout layers (input and recurrent)
  - L2 weight regularization
  - Early stopping with validation monitoring
- **Custom Review Prediction**: Test the model with your own movie reviews
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and accuracy metrics

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow 2.x / Keras
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install required packages
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

## 🚀 Usage

### Basic Training

```python
# Run the complete pipeline
python project.py
```

### Train with Different RNN Types

```python
# For LSTM (default)
histories_lstm, mean_acc_lstm = run_kfold(rnn_type='lstm', n_splits=5)

# For GRU
histories_gru, mean_acc_gru = run_kfold(rnn_type='gru', n_splits=5)

# For SimpleRNN
histories_simple, mean_acc_simple = run_kfold(rnn_type='simple', n_splits=5)
```

### Test with Custom Reviews

```python
sample_review = "The movie was absolutely amazing, I loved the story and acting."
encoded_review = encode_review(sample_review)
pred_prob = lstm_model.predict(encoded_review)[0][0]
pred_label = "Positive" if pred_prob > 0.5 else "Negative"

print(f"Prediction: {pred_label} ({pred_prob:.4f})")
```

## ⚙️ Hyperparameters

Key parameters you can adjust in the code:

```python
NUM_WORDS = 20000       # Vocabulary size
MAX_LEN = 200           # Maximum sequence length
EMBED_DIM = 128         # Embedding dimension
RNN_UNITS = 64          # RNN/LSTM/GRU units
BATCH_SIZE = 128        # Training batch size
EPOCHS = 10             # Maximum epochs (with early stopping)
N_SPLITS = 5            # K-fold cross-validation splits
L2_REG = 1e-4          # L2 regularization strength
```

## 📊 Model Architecture

### LSTM Model (Default)

```
Input (20,000 vocabulary)
    ↓
Embedding Layer (128 dims)
    ↓
Bidirectional LSTM (32 units)
    - Dropout: 0.3
    - Recurrent Dropout: 0.3
    - L2 Regularization: 0.001
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (1 unit, Sigmoid)
```

## 📈 Results

The model achieves strong performance on the IMDB test set:

- **Test Accuracy**: ~87-90%
- **Test AUC**: ~0.90-0.95
- **Mean CV Accuracy**: Consistent across folds

### Sample Output

```
Fold 1 validation accuracy: 0.8820, AUC: 0.9245
Fold 2 validation accuracy: 0.8765, AUC: 0.9198
Fold 3 validation accuracy: 0.8840, AUC: 0.9267
Fold 4 validation accuracy: 0.8792, AUC: 0.9221
Fold 5 validation accuracy: 0.8808, AUC: 0.9234
Mean val accuracy: 0.8805
```

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── project.py              # Main script with all models
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🔍 Key Functions

- `build_model()`: Constructs RNN/LSTM/GRU models with configurable parameters
- `run_kfold()`: Performs stratified k-fold cross-validation
- `train_final_and_evaluate()`: Trains final model on full dataset
- `encode_review()`: Preprocesses custom text reviews for prediction
- `plot_history()`: Visualizes training/validation curves

## 🎯 Cross-Validation Strategy

The project uses Stratified K-Fold Cross Validation to ensure:
- Balanced class distribution across folds
- Reliable performance estimates
- Reduced variance in accuracy metrics
- Better generalization assessment

## 🧪 Callbacks Used

1. **EarlyStopping**: Monitors validation loss, stops training if no improvement for 3 epochs
2. **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- IMDB dataset from Keras datasets
- TensorFlow/Keras team for the excellent deep learning framework
- Inspiration from various sentiment analysis tutorials and papers

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!

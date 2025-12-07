# AAI-590 Review Analyzer

**Capstone Project for AAI-590**  
**By:** Aditya, Deepak, and Rajesh

## Overview

The AAI-590 Review Analyzer is a comprehensive machine learning project designed to analyze product reviews and identify impactful product attributes. The system combines sentiment analysis with attribute extraction to provide insights into customer feedback for e-commerce products.

## Project Description

This project aims at analyzing product reviews to:
- **Identify impactful product attributes** such as warmth, fit, sizing, color, material, and quality
- **Perform sentiment analysis** to classify reviews as positive, neutral, or negative
- **Extract key insights** from customer feedback using rule-based and ML-based approaches
- **Provide an interactive interface** for real-time review analysis using Gradio

## Features

- **Multi-label Attribute Classification**: Identifies multiple product attributes from review text
- **Sentiment Analysis**: Uses LSTM-based deep learning model to classify review sentiment
- **Rule-based Attribute Extraction**: Implements comprehensive regex patterns for attribute detection
- **Interactive Web Interface**: Gradio-based interface for easy interaction with the models
- **Text Processing Pipeline**: Custom text preprocessing and cleaning utilities
- **Visualization**: Word cloud and other visual representations of review insights

## Repository Structure

```
AAI-590-Review-Analyzer/
├── README.md                           # Project documentation
├── AAI_590_Model_training.ipynb        # Main model training notebook for attribute extraction
├── Sentiment_Model_training1.ipynb     # LSTM sentiment analysis model training
└── Gradio_Interface.ipynb              # Interactive web interface implementation
```

## Notebooks

### 1. AAI_590_Model_training.ipynb
Main notebook for training the attribute extraction model:
- Loads and preprocesses product review datasets
- Implements helper functions (`text_helpers.py`) for text cleaning and processing
- Defines comprehensive attribute rules (`attribute_rules.py`) for various product features:
  - **Warmth**: high, low, wind_blocking
  - **Fit**: tight, small, snug, loose, rides_up, excess_length
  - **Sizing**: one_size, inconsistent
  - **Color**: accurate, off, bright_hi_vis, darker
  - **Material**: soft, itchy, thin, thick, acrylic
  - **Quality**: good, poor, durability issues
- Trains a multi-label classification model using TF-IDF and Logistic Regression
- Exports trained models for deployment

### 2. Sentiment_Model_training1.ipynb
Deep learning model for sentiment analysis:
- Processes reviews from multiple product categories (Women's Clothing, Shoe & Insole)
- Converts ratings to sentiment labels (positive, neutral, negative)
- Implements custom LSTM-based sentiment analysis model with:
  - Embedding layer
  - Multi-layer Bidirectional LSTM
  - Dropout for regularization
  - Fully connected output layers
- Trains custom tokenizer for text vectorization
- Evaluates model performance with classification metrics

### 3. Gradio_Interface.ipynb
Interactive web interface for the review analyzer:
- Loads pre-trained attribute and sentiment models
- Provides real-time review analysis through Gradio interface
- Displays:
  - Detected product attributes
  - Sentiment classification
  - Confidence scores
  - Visual representations (e.g., word clouds)
- Supports batch processing of reviews

## Technical Stack

### Libraries & Frameworks
- **Machine Learning**: scikit-learn, PyTorch
- **NLP**: TF-IDF Vectorizer, Custom Tokenizer
- **Deep Learning**: PyTorch (LSTM networks)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Interface**: Gradio
- **Utilities**: regex, joblib

### Models
1. **Attribute Classifier**: OneVsRestClassifier with Logistic Regression
2. **Sentiment Analyzer**: Multi-layer Bidirectional LSTM network
3. **Tokenizer**: Custom BPE-based tokenizer

## Setup & Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- Google Drive for model storage (if using Colab)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/asourabh-usd/AAI-590-Review-Analyzer.git
cd AAI-590-Review-Analyzer
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn wordcloud gradio joblib tokenizers
```

3. For Google Colab users:
   - Mount your Google Drive
   - Update the file paths in notebooks to match your Drive structure

## Usage

### Training Models

1. **Train Attribute Classifier**:
   - Open `AAI_590_Model_training.ipynb`
   - Run all cells to train and save the attribute extraction model
   - Models will be saved to the specified directory

2. **Train Sentiment Analyzer**:
   - Open `Sentiment_Model_training1.ipynb`
   - Run all cells to train the LSTM sentiment model
   - Tokenizer and model weights will be saved

### Using the Interface

1. Open `Gradio_Interface.ipynb`
2. Ensure trained models are available at specified paths
3. Run all cells to launch the Gradio interface
4. Enter product reviews to get:
   - Identified attributes
   - Sentiment classification
   - Visualization of key terms

## Data

The project uses product review datasets including:
- **Women's Clothing Reviews** (`Womens-clothingReviews.csv`)
- **Shoe Care & Insoles Reviews** (`shoe-care-insolesReviews.csv`)

### Dataset Features
- SourceClient
- OriginalProductName
- Title
- ReviewText
- Rating
- IsRecommended
- Category (derived)

## Model Architecture

### Sentiment Analysis LSTM
```
Input (Tokenized Text)
    ↓
Embedding Layer (vocab_size=12000, emb_dim=128)
    ↓
Bidirectional LSTM (hidden_dim=512, num_layers=5)
    ↓
Dropout (0.15)
    ↓
Fully Connected Layers
    ↓
Output (Positive/Neutral/Negative)
```

## Results & Evaluation

The models are evaluated using:
- F1-Score for multi-label attribute classification
- Classification reports for sentiment analysis
- Confusion matrices
- Custom evaluation metrics for specific attributes

## Future Enhancements

- Expand attribute categories for more product types
- Implement attention mechanisms in the LSTM model
- Add support for multi-language reviews
- Enhance visualization capabilities
- Deploy as a web service

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is part of the AAI-590 Capstone course.

## Acknowledgments

- University of San Diego - Applied Artificial Intelligence Program
- Course instructors and mentors
- Open-source community for libraries and tools

## Contact

For questions or feedback, please contact the project team:
- Aditya
- Deepak
- Rajesh

---

**Repository**: [AAI-590-Review-Analyzer](https://github.com/asourabh-usd/AAI-590-Review-Analyzer)  
**Course**: AAI-590 Capstone Project

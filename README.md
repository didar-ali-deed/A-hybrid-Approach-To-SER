# Emotion Recognition from Speech using Wav2Vec2 and Transformer Model

## Project Overview

This project focuses on creating a Speech Emotion Recognition (SER) system using **Wav2Vec2** and a **Transformer-based model** for detecting emotions from speech signals. The aim is to process raw audio data, extract relevant features, and use a neural network to classify the emotion expressed in the speech. The system utilizes multiple publicly available emotion-labeled datasets, applies feature extraction techniques using Wav2Vec2, and trains a transformer model for accurate emotion classification.

## Key Features

### 1. **Data Preprocessing**

The project integrates and preprocesses four emotion-labeled speech datasets:

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**: Contains recordings of emotional speech and song in different languages.
- **CREMA-D (The Creative Emotion Manipulation Audio Dataset)**: A large dataset with emotion-labeled speech, collected from actors.
- **TESS (Toronto Emotional Speech Set)**: A set of audio recordings of emotional speech produced by actors.
- **SAVEE (Surrey Audio-Visual Expressed Emotion)**: Includes speech data labeled with emotions.

The preprocessing involves cleaning the data, aligning and labeling it, and splitting it into training and testing sets.

### 2. **Feature Extraction using Wav2Vec2**

The project leverages the **Wav2Vec2 Transformer model** for extracting meaningful features from raw speech audio. Wav2Vec2 performs automatic feature extraction from audio files, producing high-quality representations that capture the nuances of speech signals. These features are essential for understanding the emotional content of speech.

### 3. **Transformer-Based Emotion Classification Model**

The extracted features are fed into a custom **Transformer-based model**. This model is designed to understand temporal dependencies in the audio features and perform classification tasks. It employs attention mechanisms, which help the model focus on important parts of the speech for better emotion detection.

### 4. **Emotion Classification**

The model is capable of recognizing the following emotions from speech:

- **Neutral**
- **Calm**
- **Happy**
- **Sad**
- **Angry**
- **Fear**
- **Disgust**
- **Surprise**

These emotions are classified based on the audio input processed through the trained model.

### 5. **Performance Metrics**

The system is evaluated based on accuracy, confusion matrix, and other performance metrics to ensure its effectiveness in real-world applications. The confusion matrix helps in visualizing how well the model differentiates between the different emotions.

## File Structure

Here’s an overview of the project’s directory structure:

├── Preprocessed Data/ │ └── combined_emotions.csv # Combined dataset after preprocessing ├── Extracted Features/ │ └── wav2vec_features.csv # Extracted features using Wav2Vec2 ├── Models/ │ └── emotion_transformer_model.pth # Saved model after training ├── Results/ │ └── confusion_matrix.png # Confusion matrix visualization └── Logs/ └── training_log.txt # Log file for training process

### Directory Breakdown:

- **Preprocessed Data**: Contains the final dataset after combining the emotion-labeled speech data from multiple sources.
- **Extracted Features**: Stores the CSV file containing the features extracted from speech using Wav2Vec2.
- **Models**: Contains the trained emotion classification model.
- **Results**: Includes evaluation outputs such as confusion matrices or other visualizations.
- **Logs**: Stores logs generated during the training and evaluation processes.

## Installation Requirements

This project requires Python 3.7+ and the following Python libraries:

- `transformers`: Hugging Face’s transformer models, including Wav2Vec2.
- `librosa`: A package for analyzing audio and extracting features.
- `torch`: The deep learning framework for model training.
- `pandas`: A library for data manipulation.
- `scikit-learn`: For model evaluation and metrics.
- `seaborn`: For data visualization.
- `matplotlib`: For generating plots.
- `tqdm`: A library for displaying progress bars during processing.

To install the necessary dependencies, run the following command:

```bash
pip install transformers librosa torch pandas scikit-learn seaborn matplotlib tqdm
```

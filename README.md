# Emotion Recognition from Speech using Wav2Vec2 and Transformer Model

## Project Overview

This project builds a **Speech Emotion Recognition (SER)** system using **Wav2Vec2** and a **Transformer-based model** to detect emotions from speech signals. It processes raw audio data, extracts meaningful features, and trains a deep learning model to classify emotions. By integrating multiple publicly available emotion-labeled datasets, the system achieves robust and diverse training data for real-world applicability.

## Key Features

### 1. **Data Preprocessing**

The project integrates and preprocesses data from four prominent emotion-labeled speech datasets:

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**: Features recordings of emotional speech and song in various languages.
- **CREMA-D (The Creative Emotion Manipulation Audio Dataset)**: A comprehensive dataset of emotion-labeled speech from professional actors.
- **TESS (Toronto Emotional Speech Set)**: A collection of audio recordings of emotional speech by professional speakers.
- **SAVEE (Surrey Audio-Visual Expressed Emotion)**: Contains speech data labeled with emotional categories.

Preprocessing steps include:

- Cleaning and standardizing the data.
- Aligning datasets for consistency.
- Splitting data into training and testing sets.

### 2. **Feature Extraction using Wav2Vec2**

The **Wav2Vec2 Transformer model** is employed for feature extraction. Wav2Vec2 processes raw speech audio and produces high-quality representations that capture speech nuances essential for emotion detection. This step reduces the dependency on handcrafted features and improves accuracy.

### 3. **Transformer-Based Emotion Classification Model**

The extracted features are input into a custom **Transformer-based model** designed to analyze temporal dependencies. The model leverages attention mechanisms to focus on critical parts of the speech signal, enhancing emotion recognition accuracy. The model is tailored for performance and scalability.

### 4. **Emotion Classification**

The system recognizes the following emotions from speech data:

- **Neutral**
- **Calm**
- **Happy**
- **Sad**
- **Angry**
- **Fear**
- **Disgust**
- **Surprise**

### 5. **Performance Metrics**

The system is evaluated using:

- **Accuracy**: Measures overall performance.
- **Confusion Matrix**: Visualizes classification performance by displaying true positives and false positives for each class.
- **Precision, Recall, and F1 Score**: Provides detailed insights into model performance for each emotion category.

## File Structure

The project is organized as follows:

├── Preprocessed Data/ │ └── combined_emotions.csv # Combined dataset after preprocessing ├── Extracted Features/ │ └── wav2vec_features.csv # Extracted features using Wav2Vec2 ├── Models/ │ └── emotion_transformer_model.pth # Trained emotion classification model ├── Results/ │ └── confusion_matrix.png # Confusion matrix visualization ├── Logs/ │ └── training_log.txt # Log file for the training process

### Directory Breakdown:

- **Preprocessed Data**: Contains the processed dataset after combining emotion-labeled data.
- **Extracted Features**: Stores the extracted features from Wav2Vec2.
- **Models**: Contains the trained emotion classification model.
- **Results**: Includes evaluation outputs like confusion matrices.
- **Logs**: Stores logs generated during training and evaluation.

## Installation Requirements

This project requires **Python 3.7+** and the following Python libraries:

- `transformers`: For Wav2Vec2 and other Hugging Face models.
- `librosa`: For audio analysis and feature extraction.
- `torch`: Deep learning framework for training the model.
- `pandas`: For data manipulation and preprocessing.
- `scikit-learn`: For model evaluation and performance metrics.
- `seaborn`: For data visualization.
- `matplotlib`: For generating visualizations like confusion matrices.
- `tqdm`: For displaying progress bars during processing.

### Install Dependencies

Run the following command to install the required libraries:

pip install transformers librosa torch pandas scikit-learn seaborn matplotlib tqdm

### Usage

1. **Data Preprocessing**
   Run the preprocessing script to clean and combine datasets:

python scripts/preprocess_data.py 2. **Feature Extraction**
Extract features using Wav2Vec2 by running the feature extraction script:

python scripts/extract_features.py 3. **Model Training**
Train the emotion classification model using the following command:

python scripts/train_emotion_classifier.py 4. **Evaluation**
Evaluate the trained model and generate performance metrics:

python scripts/evaluate_model.py 5. **Visualization**
View the confusion matrix and other visualizations in the Results/ directory.

### Conclusion

This project demonstrates the effectiveness of combining Wav2Vec2 and a Transformer-based model for speech emotion recognition. It provides a robust framework for processing audio data, extracting features, and classifying emotions with high accuracy. By leveraging modern deep learning techniques, the system sets the stage for improved human-computer interaction and applications in domains like customer service, healthcare, and virtual assistants

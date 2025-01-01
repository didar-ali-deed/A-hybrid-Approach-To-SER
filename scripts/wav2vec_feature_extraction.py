from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import pandas as pd
import os
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore specific UserWarnings

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Path to preprocessed dataset
data_path = "../Preprocessed Data/combined_emotions.csv"
output_path = "../Extracted Features/wav2vec_features.csv"

# Ensure output directory exists
if not os.path.exists("../Extracted Features"):
    os.makedirs("../Extracted Features")

# Function to extract Wav2Vec2.0 features
def extract_wav2vec_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features.mean(dim=1).squeeze().numpy()  # Mean pooling

# Load preprocessed data
df = pd.read_csv(data_path)

# Group the dataset by emotion and limit to first 100 samples for each emotion
df_limited = df.groupby('Emotions').head(50)

# Iterate over dataset and extract Wav2Vec features
features = []
labels = []

for i, row in tqdm(df_limited.iterrows(), total=df_limited.shape[0], desc="Extracting Wav2Vec features"):
    file_path = row['Path']
    emotion = row['Emotions']
    try:
        wav2vec_features = extract_wav2vec_features(file_path)
        features.append(wav2vec_features)
        labels.append(emotion)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save features to CSV
features_df = pd.DataFrame(features)
features_df['label'] = labels
features_df.to_csv(output_path, index=False)

print(f"Wav2Vec features saved to {output_path}")

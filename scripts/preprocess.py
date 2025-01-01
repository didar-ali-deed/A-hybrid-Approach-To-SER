import os
import pandas as pd

# Define paths to your datasets
RAVDESS = "../data/ravdess/"
CREMA = "../data/crema-d/AudioWAV/"
TESS = "../data/tess/TESS Toronto emotional speech set data/"
SAVEE = "../data/savee/ALL/"

# Emotion mapping for consistency across datasets
emotion_map = {
    'neutral': 'neutral', 'calm': 'calm', 'happy': 'happy', 'sad': 'sad', 
    'angry': 'angry', 'fear': 'fear', 'disgust': 'disgust', 'surprise': 'surprise'
}

# RAVDESS Dataset
def process_ravdess():
    ravdess_directory_list = os.listdir(RAVDESS)
    file_emotion = []
    file_path = []
    
    for actor_dir in ravdess_directory_list:
        actor_path = os.path.join(RAVDESS, actor_dir)
        if os.path.isdir(actor_path):
            actor_files = os.listdir(actor_path)
            for file in actor_files:
                if file.endswith(".wav"):
                    part = file.split('.')[0].split('-')
                    if len(part) >= 3:
                        emotion_code = int(part[2])
                        emotion = {
                            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 
                            6: 'fear', 7: 'disgust', 8: 'surprise'
                        }.get(emotion_code, 'unknown')
                        file_emotion.append(emotion_map.get(emotion, emotion))
                        file_path.append(os.path.join(actor_path, file))
    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})

# CREMA-D Dataset
def process_crema():
    crema_directory_list = os.listdir(CREMA)
    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        if file.endswith(".wav"):
            file_path.append(os.path.join(CREMA, file))
            part = file.split('_')[2]
            emotion = {
                'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 
                'HAP': 'happy', 'NEU': 'neutral', 'SUR': 'surprise'
            }.get(part, 'unknown')
            file_emotion.append(emotion_map.get(emotion, emotion))
    
    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})

# TESS Dataset
def process_tess():
    tess_directory_list = os.listdir(TESS)
    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        emotion_folder = os.path.join(TESS, dir)
        if os.path.isdir(emotion_folder):
            files = os.listdir(emotion_folder)
            for file in files:
                if file.endswith(".wav"):
                    emotion = dir.split('_')[-1].lower()
                    file_emotion.append(emotion_map.get(emotion, emotion))
                    file_path.append(os.path.join(emotion_folder, file))
    
    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})

# SAVEE Dataset
def process_savee():
    savee_directory_list = os.listdir(SAVEE)
    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        if file.endswith(".wav"):
            file_path.append(os.path.join(SAVEE, file))
            part = file.split('_')[1][:-6]
            emotion = {
                'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy',
                'n': 'neutral', 'sa': 'sad'
            }.get(part, 'surprise')
            file_emotion.append(emotion_map.get(emotion, emotion))
    
    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})

# Main function to combine all datasets
def combine_datasets():
    # Process each dataset
    ravdess_df = process_ravdess()
    crema_df = process_crema()
    tess_df = process_tess()
    savee_df = process_savee()

    # Combine all datasets into a single dataframe
    combined_df = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis=0, ignore_index=True)

    # Ensure the output directory exists
    output_dir = "../Preprocessed Data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(os.path.join(output_dir, "combined_emotions.csv"), index=False)
    print(f"Combined dataset saved to {os.path.join(output_dir, 'combined_emotions.csv')}")

if __name__ == "__main__":
    combine_datasets()

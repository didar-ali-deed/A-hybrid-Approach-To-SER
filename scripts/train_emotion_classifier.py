import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define Transformer-based model for emotion classification
class EmotionTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, output_dim):
        super(EmotionTransformer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads), num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        transformer_out = self.transformer(x)
        pooled_out = transformer_out.mean(dim=1)  # Global average pooling
        return self.fc(pooled_out)

# Custom dataset to load Wav2Vec features
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, features_file):
        self.data = pd.read_csv(features_file)
        self.X = self.data.iloc[:, :-1].values  # All columns except the last one (features)
        self.y = self.data.iloc[:, -1].values   # Last column (labels)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Parameters
input_dim = 768  # Wav2Vec feature dimension
num_heads = 8  # Number of attention heads in the transformer
num_layers = 3  # Number of transformer layers
output_dim = 8  # Number of emotion classes
batch_size = 64
learning_rate = 0.0001
epochs = 50

# Load dataset
features_file = "../Extracted Features/wav2vec_features.csv"
dataset = EmotionDataset(features_file)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer, and Scheduler
model = EmotionTransformer(input_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Directories for saving models and logs
model_dir = "../models/"
log_dir = "../logs/"
result_dir = "../results/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Training and Validation Functions
def validate_model():
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return avg_loss, f1

def train_model():
    best_val_loss = float('inf')
    train_log = open(os.path.join(log_dir, "training_log.txt"), "w")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        avg_train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_loss, val_f1 = validate_model()

        # Logging
        log_entry = f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}\n"
        train_log.write(log_entry)
        print(log_entry.strip())

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

        scheduler.step()

    train_log.close()

# Evaluation Function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save confusion matrix
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(result_dir, 'confusion_matrix.png')}")
    plt.show()

if __name__ == "__main__":
    train_model()
    evaluate_model()

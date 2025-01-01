import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths for saving models and results
base_dir = "../"
model_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")
logs_dir = os.path.join(base_dir, "logs")

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Define paths
model_save_path = os.path.join(model_dir, "emotion_transformer_model.pth")
confusion_matrix_save_path = os.path.join(results_dir, "confusion_matrix.png")
results_file = os.path.join(results_dir, "results.txt")
log_file = os.path.join(logs_dir, "training_log.txt")

# Transformer-based Model for Emotion Classification
class EmotionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, nhead=8, dim_feedforward=512):
        super(EmotionTransformer, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean pooling
        x = self.layer_norm(x)
        out = self.fc(x)
        return out

# Custom dataset class to load features
class EmotionDataset(Dataset):
    def __init__(self, features_file):
        self.data = pd.read_csv(features_file)
        self.X = self.data.iloc[:, :-1].values  # Features
        self.y = self.data.iloc[:, -1].values  # Labels
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Hyperparameters
input_dim = 768
hidden_dim = 256
output_dim = 8
batch_size = 64
learning_rate = 0.000001
epochs = 100
num_layers = 4
nhead = 8
patience = 10

# Load dataset
features_file = "../Extracted Features/wav2vec_features.csv"
dataset = EmotionDataset(features_file)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
model = EmotionTransformer(input_dim, hidden_dim, output_dim, num_layers=num_layers, nhead=nhead)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Validation function
def validate_model():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Save the model
def save_model():
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Training loop with early stopping
def train_model():
    early_stop_counter = 0
    best_val_loss = float('inf')

    with open(log_file, "w") as log:
        log.write("Starting training loop...\n")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            val_loss = validate_model()

            log.write(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
                      f"Accuracy: {accuracy:.2f}%, Val Loss: {val_loss:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_model()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break

            scheduler.step()

# Evaluation and confusion matrix
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
    with open(results_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")

    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(confusion_matrix_save_path)
    plt.show()

# Main Execution
if __name__ == "__main__":
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Model loaded from {model_save_path}")
    else:
        train_model()

    evaluate_model()

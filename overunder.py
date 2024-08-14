import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# Load the dataset
merged_dataset = pd.read_csv("nfl_merged_final.csv")

# Clean the dataset
data_cleaned = merged_dataset.dropna()

def add_team_identifier(df):
    df.loc[:, 'Team'] = df[[col for col in df.columns if col.startswith('Team_')]].idxmax(axis=1)
    df['Team'] = df['Team'].str.replace('Team_', '')  # Remove the 'Team_' prefix

    df.loc[:, 'Opp'] = df[[col for col in df.columns if col.startswith('Opp_')]].idxmax(axis=1)
    df['Opp'] = df['Opp'].str.replace('Opp_', '')  # Remove the 'Opp_' prefix
    return df

# Add the team and opponent identifiers
data_cleaned = add_team_identifier(data_cleaned)

def calculate_rolling_averages(df, team_column, points_column, n_games=5):
    df.loc[:, f'Avg_{points_column}_Last_{n_games}'] = df.groupby(team_column)[points_column].transform(
        lambda x: x.rolling(window=n_games, min_periods=1).mean()
    )
    return df

# Calculate rolling averages
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', 'Off_Pts', n_games=5)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Opp', 'Def_Pts', n_games=5)

team_columns = [col for col in data_cleaned.columns if col.startswith('Team_') or col.startswith('Opp_')]
data_cleaned['Over_Under_Target'] = (data_cleaned['Off_Pts'] + data_cleaned['Def_Pts']) > data_cleaned['Total']
data_cleaned['Over_Under_Target'] = data_cleaned['Over_Under_Target'].astype(int)

# Feature set
X = data_cleaned[['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5'] + team_columns]
y = data_cleaned['Over_Under_Target']

# Shuffle and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
test_indices = X_test.index

# Define the neural network class
class OverUnderNN(nn.Module):
    def __init__(self, input_size):
        super(OverUnderNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        output = torch.sigmoid(self.output_layer(x))  # Sigmoid for binary classification
        return output

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
model = OverUnderNN(input_size=input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Cross-Validation
print("\nPerforming Cross-Validation...")
skf = StratifiedKFold(n_splits=5)
cross_val_acc = []
for train_index, val_index in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    X_train_cv_tensor = torch.tensor(X_train_cv.values, dtype=torch.float32)
    y_train_cv_tensor = torch.tensor(y_train_cv.values, dtype=torch.float32).view(-1, 1)

    X_val_cv_tensor = torch.tensor(X_val_cv.values, dtype=torch.float32)
    y_val_cv_tensor = torch.tensor(y_val_cv.values, dtype=torch.float32).view(-1, 1)

    model_cv = OverUnderNN(input_size=input_size)
    optimizer_cv = optim.Adam(model_cv.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model_cv.train()
        predictions_cv = model_cv(X_train_cv_tensor)
        loss_cv = criterion(predictions_cv, y_train_cv_tensor)
        optimizer_cv.zero_grad()
        loss_cv.backward()
        optimizer_cv.step()

    model_cv.eval()
    with torch.no_grad():
        predictions_cv = model_cv(X_val_cv_tensor)
        predicted_classes_cv = (predictions_cv > 0.5).float()
        accuracy_cv = accuracy_score(y_val_cv, predicted_classes_cv)
        cross_val_acc.append(accuracy_cv)

print(f"Cross-Validation Accuracy Scores: {cross_val_acc}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cross_val_acc):.4f}")

# Test model on test data
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, predicted_classes)

print(f'Test Loss: {test_loss.item():.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Permutation test (sanity check)
print("\nPerforming Permutation Test...")
y_permuted = np.random.permutation(y)
X_train_perm, X_test_perm, y_train_perm, y_test_perm = train_test_split(X, y_permuted, test_size=0.3, random_state=42, shuffle=True)

X_train_perm_tensor = torch.tensor(X_train_perm.values, dtype=torch.float32)
y_train_perm_tensor = torch.tensor(y_train_perm, dtype=torch.float32).view(-1, 1)  # No .values here

X_test_perm_tensor = torch.tensor(X_test_perm.values, dtype=torch.float32)
y_test_perm_tensor = torch.tensor(y_test_perm, dtype=torch.float32).view(-1, 1)  # No .values here

# Train on permuted data
model_perm = OverUnderNN(input_size=input_size)
optimizer_perm = optim.Adam(model_perm.parameters(), lr=0.001)

for epoch in range(epochs):
    model_perm.train()
    predictions_perm = model_perm(X_train_perm_tensor)
    loss_perm = criterion(predictions_perm, y_train_perm_tensor)
    optimizer_perm.zero_grad()
    loss_perm.backward()
    optimizer_perm.step()

# Evaluate on permuted data
model_perm.eval()
with torch.no_grad():
    predictions_perm = model_perm(X_test_perm_tensor)
    predicted_classes_perm = (predictions_perm > 0.5).float()
    accuracy_perm = accuracy_score(y_test_perm_tensor, predicted_classes_perm)

print(f'Permutation Test Accuracy: {accuracy_perm:.4f}')

# Manually inspect predictions
print("\nManual Inspection of Predictions:")
for i, idx in enumerate(test_indices[:10]):
    actual_score = data_cleaned.loc[idx, 'Off_Pts'] + data_cleaned.loc[idx, 'Def_Pts']
    over_under_line = data_cleaned.loc[idx, 'Total']
    predicted = "Over" if predicted_classes[i].item() == 1 else "Under"
    actual_result = "Over" if y_test.iloc[i] == 1 else "Under"
    print(f"Game {i + 1}: Predicted: {predicted}, Actual: {actual_result}, Actual Score: {actual_score}, Over/Under Line: {over_under_line}")


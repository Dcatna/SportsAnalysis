import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from model import OverUnderNN

# Load the dataset
merged_dataset = pd.read_csv("nfl_merged_corrected.csv")
player_data = pd.read_csv("all_player_data_cleaned.csv")

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

# Combine Season and Date to create a full date
data_cleaned['Full_Date'] = data_cleaned['Season'].astype(str) + ' ' + data_cleaned['Date']

# Convert to datetime using the correct format
data_cleaned['Full_Date'] = pd.to_datetime(data_cleaned['Full_Date'], format='%Y %B %d')

# If you need the date as a string formatted as 'YYYY-MM-DD'
data_cleaned['Date'] = data_cleaned['Full_Date'].dt.strftime('%Y-%m-%d')

# Now you can drop the 'Full_Date' column if it's no longer needed
data_cleaned = data_cleaned.drop(columns=['Full_Date'])

# Standardize team names
team_mapping = {
    'CRD': 'ARI', 'RAM': 'STL', 'Yds': 'LA', 'SDG': 'LAC',
    'RAI': 'OAK', 'HTX': 'HOU', 'OTI': 'TEN', 'Yds_Lost_Sacks': 'LVR'
}
team_name_mapping = {
    'SDG': 'LAC', 'STL': 'LA', 'OAK': 'LVR', 'RAM': 'LA',
    'PHO': 'ARI', 'CRD': 'ARI', 'RAI': 'LVR', 'HTX': 'HOU',
    'CLT': 'IND', 'KAN': 'KC', 'GN': 'NYG', 'NYG': 'NYG',
    'NYJ': 'NYJ', 'JAC': 'JAX', 'NOR': 'NO', 'TAM': 'TB', 'WAS': 'WAS',
    'GB': 'GNB', 'NE': 'NWE', 'SF': 'SFO', 'TEN': 'TEN'
}
player_data['Tm'] = player_data['Tm'].map(team_name_mapping)
print("jkl;jkl", data_cleaned['Team'].unique())
print("Standardized Teams in player_data:", player_data['Tm'].unique())
missing_teams = player_data['Tm'].isna().sum()
print(f"Number of missing teams after mapping: {missing_teams}")

# Handling missing teams
if missing_teams > 0:
    print("Handling missing teams...")
    player_data['Tm'] = player_data['Tm'].fillna('Unknown')
    unmapped_teams = player_data[player_data['Tm'] == 'Unknown']['Player'].unique()
    print(f"Unmapped Teams: {unmapped_teams}")
else:
    unmapped_teams = []

# Apply the mapping to data_cleaned
data_cleaned['Team'] = data_cleaned['Team'].replace(team_mapping)

# Convert Time_of_Possession to seconds
def convert_time_of_possession_to_seconds(time_str):
    if pd.isna(time_str) or ':' not in time_str:
        return 0  
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        return 0 

n_games = 5
# Apply the conversion
data_cleaned['Time_of_Possession_Seconds'] = data_cleaned['Time_of_Possession'].apply(convert_time_of_possession_to_seconds)

# Function to calculate rolling averages
def calculate_rolling_averages(df, team_column, points_column, n_games=5):
    df.loc[:, f'Avg_{points_column}_Last_{n_games}'] = df.groupby(team_column)[points_column].transform(
        lambda x: x.rolling(window=n_games, min_periods=1).mean()
    )
    return df

# Calculate rolling averages on key metrics
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', 'Time_of_Possession_Seconds', n_games=n_games)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', 'Off_Pts', n_games=n_games)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Opp', 'Def_Pts', n_games=n_games)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', '3DConv', n_games=n_games)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', '4DConv', n_games=n_games)

# Ensure the 'Over_Under_Target' column is created before filtering columns
data_cleaned['Over_Under_Target'] = (data_cleaned['Off_Pts'] + data_cleaned['Def_Pts']) > data_cleaned['Total']
data_cleaned['Over_Under_Target'] = data_cleaned['Over_Under_Target'].astype(int)

# Drop the specific columns you don’t want
columns_to_drop = ['Team_Pts', 'Opp_Pts', 'Team_Yds', 'Team_Yds_Lost_Sacks']
data_cleaned = data_cleaned.drop(columns=columns_to_drop, errors='ignore')

# One-hot encode players
player_columns = player_data['Player'].unique()

for player in player_columns:
    player_data[player] = player_data['Player'].apply(lambda x: 1 if x == player else 0)

# Group by Date and Tm (team) and sum the player columns
one_hot_encoded_data = player_data.groupby(['Date', 'Tm'])[player_columns].sum().reset_index()

# Merge the player data with the cleaned game data
data_cleaned = pd.merge(data_cleaned, one_hot_encoded_data, how='left', left_on=['Date', 'Team'], right_on=['Date', 'Tm'])

# Drop the redundant 'Tm' column after merging
data_cleaned.drop(columns=['Tm'], inplace=True, errors='ignore')

# Replace NaN values with 0 in player columns
data_cleaned[player_columns] = data_cleaned[player_columns].fillna(0)

team_columns = [col for col in data_cleaned.columns if col.startswith('Team_') or col.startswith('Opp_')]
player_data['Date'] = pd.to_datetime(player_data['Date'])
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

# Ensure team names are consistent (e.g., all uppercase)
player_data['Tm'] = player_data['Tm'].str.upper()
data_cleaned['Team'] = data_cleaned['Team'].str.upper()

# Extract unique dates and teams from data_cleaned
cleaned_unique_dates = data_cleaned['Date'].unique()
cleaned_unique_teams = data_cleaned['Team'].unique()

# After one-hot encoding and before merging
print("One-Hot Encoded Data Sample:")
print(one_hot_encoded_data.head())

# After merging and filling NaNs
print("Final Data with Player Columns (Sample):")
print(data_cleaned[player_columns].head())

print("NaN Values in Player Columns After Filling:")
print(data_cleaned[player_columns].isna().sum())

# Filter and inspect specific games or players
specific_game = data_cleaned[(data_cleaned['Date'] == '2014-09-04') & (data_cleaned['Team'] == 'SEA')]
print("Specific Game Player Encoding:")
print(specific_game[player_columns].head())

# Inspect a few specific player columns across the first few rows
specific_players = ['Philip Rivers', 'Ryan Mathews', 'Danny Woodhead', 'Antonio Gates']
print("First few rows for specific players:")
print(data_cleaned[specific_players].head(10))

# Print all player columns for the first row
print("All player columns for the first row:")
print(data_cleaned.loc[0, player_columns])

# Prepare the data for training
X = data_cleaned[['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5', 
                  'Avg_Time_of_Possession_Seconds_Last_5', 'Avg_3DConv_Last_5', 'Avg_4DConv_Last_5'] + team_columns + list(player_columns)]
y = data_cleaned['Over_Under_Target']
data_cleaned.to_csv("cleaned_data_for_inspection.csv", index=False)
# Shuffle and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
test_indices = X_test.index

# Define the neural network class

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]
print("input size: ", input_size)
model = OverUnderNN(input_size=input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Training loop
epochs = 2000
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
y_train_perm_tensor = torch.tensor(y_train_perm, dtype=torch.float32).view(-1, 1)

X_test_perm_tensor = torch.tensor(X_test_perm.values, dtype=torch.float32)
y_test_perm_tensor = torch.tensor(y_test_perm, dtype=torch.float32).view(-1, 1)

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

# Save the model
try:
    torch.save(model.state_dict(), "overunder_model.pth")
    print("MODEL SAVED")
except Exception as e:
    print(f"Error saving model: {e}")

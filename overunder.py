import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
merged_dataset = pd.read_csv("nfl_pts_and_vegas_with_teams.csv")

# Clean the dataset
data_cleaned = merged_dataset.dropna()
# Assuming data_cleaned is your DataFrame containing the one-hot encoded teams

# Assuming data_cleaned is your DataFrame containing the one-hot encoded teams
def add_team_identifier(df):
    # Create a new column 'Team' by finding which one-hot encoded column has a value of 1
    df.loc[:, 'Team'] = df[[col for col in df.columns if col.startswith('Team_')]].idxmax(axis=1)
    df['Team'] = df['Team'].str.replace('Team_', '')  # Remove the 'Team_' prefix

    df.loc[:, 'Opp'] = df[[col for col in df.columns if col.startswith('Opp_')]].idxmax(axis=1)
    df['Opp'] = df['Opp'].str.replace('Opp_', '')  # Remove the 'Opp_' prefix

    return df

# Add the team and opponent identifiers
data_cleaned = add_team_identifier(data_cleaned)

# Now you can calculate the rolling averages using the new 'Team' column
def calculate_rolling_averages(df, team_column, points_column, n_games=5):
    df.loc[:, f'Avg_{points_column}_Last_{n_games}'] = df.groupby(team_column)[points_column].transform(
        lambda x: x.rolling(window=n_games, min_periods=1).mean()
    )
    return df

# Calculate rolling averages for both the team and the opponent
data_cleaned = calculate_rolling_averages(data_cleaned, 'Team', 'Off_Pts', n_games=5)
data_cleaned = calculate_rolling_averages(data_cleaned, 'Opp', 'Def_Pts', n_games=5)
# Now, data_cleaned contains new columns for these averages:
# 'Avg_Off_Pts_Last_5' for the team's average offensive points over the last 5 games
# 'Avg_Def_Pts_Last_5' for the opponent's average defensive points allowed over the last 5 games
print(data_cleaned)
# You can now include these averages in your feature set (X)
team_columns = [col for col in data_cleaned.columns if col.startswith('Team_') or col.startswith('Opp_')]

# Update the feature set (X) to include the average scoring columns and the one-hot encoded team columns
X = data_cleaned[['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5'] + team_columns]

# Your target variable (y) remains the same
y = data_cleaned['Off_Pts'] + data_cleaned['Def_Pts']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
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
        output = self.output_layer(x)
        return output

# Initialize the model
input_size = X_train.shape[1]
model = OverUnderNN(input_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert the training and test data to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Lists to track losses
train_losses = []
val_losses = []

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    train_losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor)
        val_loss = criterion(val_predictions, y_test_tensor)
        val_losses.append(val_loss.item())

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plotting the training and validation losses
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss over Epochs')
# plt.show()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss (MSE): {test_loss.item():.4f}')

    # Convert tensors to numpy arrays for additional metrics
    predictions_numpy = predictions.numpy().flatten()
    y_test_numpy = y_test_tensor.numpy().flatten()

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_numpy, predictions_numpy)
    print(f'Test MAE: {mae:.4f}')

    # Calculate R² Score
    r2 = r2_score(y_test_numpy, predictions_numpy)
    print(f'R² Score: {r2:.4f}')

predictions = predictions.round()
for i in range(10):
    print(f"Game {i + 1}: Predicted Total: {predictions[i][0]:.2f}, Actual Total: {y_test.values[i]:.2f}")

try:
    torch.save(model.state_dict(), 'over_under_model.pth')
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving the model: {e}")

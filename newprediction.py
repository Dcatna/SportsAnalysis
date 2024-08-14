import torch
import pandas as pd
from overunder import OverUnderNN

# Load the trained model
model_path = 'over_under_model.pth'
input_size = 8 + len(['Team_A', 'Team_B', ...])  # Adjust based on your training data (number of features + one-hot encoded teams)
model = OverUnderNN(input_size=input_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Manually input the game data
new_game_data = {
    'Spread': -3.5,
    'Total': 47.5,
    'Home': 1,
    'Off_Pts': 0,  # Set to 0 if you're predicting this
    'Def_Pts': 0,  # Set to 0 if you're predicting this
    'Point_Diff': 0,  # Set to 0 since it's unknown before the game
    'Spread_Covered': 0,  # Set to 0 or omit if calculating post-game
    'Actual_vs_Expected': 0.0  # Set to 0 since it's unknown before the game
}

# One-Hot Encode the teams (manually)
teams = ['Team_A', 'Team_B', ...]  # Replace with all team names used in training

for team in teams:
    new_game_data[f'Team_{team}'] = 1 if team == 'Team_A' else 0  # Replace 'Team_A' with the home team
    new_game_data[f'Opp_{team}'] = 1 if team == 'Team_B' else 0  # Replace 'Team_B' with the opponent

# Convert the data to a DataFrame (for consistency)
input_df = pd.DataFrame([new_game_data])

# Extract the feature columns (ensure order matches training)
X_new = input_df[['Spread', 'Total', 'Home', 'Off_Pts', 'Def_Pts', 'Point_Diff', 'Spread_Covered', 'Actual_vs_Expected'] + [f'Team_{team}' for team in teams] + [f'Opp_{team}' for team in teams]]

# Convert the DataFrame to a PyTorch tensor
X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    prediction = model(X_new_tensor)

# Output the predicted total points for the game
predicted_total = prediction.item()
print(f'Predicted Total Points: {predicted_total:.2f}')

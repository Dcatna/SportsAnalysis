import torch
import pandas as pd
from model import OverUnderNN

# Load the trained model
model_path = 'overunder_model.pth'
input_size = 73
model = OverUnderNN(input_size=input_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Manually input the game data with the necessary features
new_game_data = {
    'Spread': 2.5,
    'Total': 36,
    'Home': 0,
    'Avg_Off_Pts_Last_5': 84,
    'Avg_Def_Pts_Last_5': 99
}

# Predefined team columns from the training data
team_columns = [
    'Team_ATL', 'Team_BUF', 'Team_CAR', 'Team_CHI', 'Team_CIN', 'Team_CLE', 'Team_CLT', 'Team_CRD', 'Team_DAL', 
    'Team_DEN', 'Team_DET', 'Team_GNB', 'Team_HTX', 'Team_JAX', 'Team_KAN', 'Team_MIA', 'Team_MIN', 'Team_NOR', 
    'Team_NWE', 'Team_NYG', 'Team_NYJ', 'Team_OTI', 'Team_PHI', 'Team_PIT', 'Team_RAI', 'Team_RAM', 'Team_RAV', 
    'Team_SDG', 'Team_SEA', 'Team_SFO', 'Team_TAM', 'Team_WAS', 'Opp_ATL', 'Opp_BUF', 'Opp_CAR', 'Opp_CHI', 
    'Opp_CIN', 'Opp_CLE', 'Opp_CLT', 'Opp_CRD', 'Opp_DAL', 'Opp_DEN', 'Opp_DET', 'Opp_GNB', 'Opp_HTX', 'Opp_JAX', 
    'Opp_KAN', 'Opp_MIA', 'Opp_MIN', 'Opp_NOR', 'Opp_NWE', 'Opp_NYG', 'Opp_NYJ', 'Opp_OTI', 'Opp_PHI', 'Opp_PIT', 
    'Opp_RAI', 'Opp_RAM', 'Opp_RAV', 'Opp_SDG', 'Opp_SEA', 'Opp_SFO', 'Opp_TAM', 'Opp_WAS'
]

# Initialize the one-hot encoding columns with zeros
for col in team_columns:
    new_game_data[col] = 0

# Set the actual teams for this game
new_game_data['Team_ATL'] = 1  # Example: Home team is ATL
new_game_data['Opp_BUF'] = 1   # Example: Opponent is BUF

# Convert to DataFrame
input_df = pd.DataFrame([new_game_data])

# Ensure the DataFrame columns are in the correct order
expected_columns = ['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5'] + team_columns

# Reorder the DataFrame to match the expected column order
X_new = input_df[expected_columns]

# Print the columns used in prediction input
print("Columns in Prediction Input:")
for col in X_new.columns:
    print(col)

# Print the final number of features to confirm
print(f"\nFinal number of features in input: {X_new.shape[1]}")

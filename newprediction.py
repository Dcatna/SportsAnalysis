import torch
import pandas as pd
from model import OverUnderNN

# Load the trained model
model_path = 'overunder_model.pth'
data = pd.read_csv('t.csv', )
input_size = 1971
model = OverUnderNN(input_size=input_size)
model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

team_columns = [col for col in data.columns if col.startswith('Team_') or col.startswith('Opp_')]
player_columns = [col for col in data.columns if col.startswith('player_')]
day_columns = [col for col in data.columns if col.startswith('Day_')]
year_columns = [col for col in data.columns if col.startswith('Year_')]
new_game_data = {
    'Spread': 6.0,
    'Total': 45.0,
    'Home': 1,
    'Team': 'CHI',
    'Opponent': 'HTX',
    # 'Date': 'September 8',
    # 'Season': '2024',

}

def encode_teams(new_game_data):
    # One-hot encoding the teams based on existing columns
    teams = [col for col in data.columns if col.startswith('Team_') or col.startswith('Opp_')]
    
    for team_col in teams:
        # Example: Team_PHI becomes 1 if it's Philadelphia, else 0
        if team_col == f'Team_{new_game_data["Team"]}':
            new_game_data[team_col] = 1
        elif team_col == f'Opp_{new_game_data["Opponent"]}':
            new_game_data[team_col] = 1
        else:
            new_game_data[team_col] = 0
    
    return new_game_data

def get_last_game_player_data(new_game_data):
    player_columns = [col for col in data.columns if col.startswith('player_')]
    # Sort by team and date to get the last game
    all_data_sorted = data.sort_values(by=['Team', 'Date'], ascending=[True, False])
    last_game_players = all_data_sorted.groupby('Team').first().reset_index()

    # Get the last game player data for the specified team and opponent
    team_player_data = last_game_players[last_game_players['Team'] == new_game_data['Team']]
    opp_player_data = last_game_players[last_game_players['Team'] == new_game_data['Opponent']]
    
    # Add player columns to the new game data
    for player_col in player_columns:
        new_game_data[player_col] = team_player_data[player_col].values[0]
        new_game_data[player_col] = opp_player_data[player_col].values[0]
    
    return new_game_data

def encode_day_and_year(new_game_data):
    # Example encoding day and year
    day_columns = [col for col in data.columns if col.startswith('Day_')]
    day_of_game = 'Sunday'  # Change this to match the actual day
    year_of_game = 2024      # Change this to the actual year
    
    # Manually map days to columns, example:
    day_mapping = {'Monday': 'Day_Monday', 'Sunday': 'Day_Sunday', 'Thursday': 'Day_Thursday'}
    year_columns = [col for col in data.columns if col.startswith('Year_')]
    
    # Set day columns
    for day_col in day_columns:
        new_game_data[day_col] = 1 if day_mapping[day_of_game] == day_col else 0
    
    # Set year columns (one-hot encode)
    for year_col in year_columns:
        new_game_data[year_col] = 1 if str(year_of_game) in year_col else 0
    
    return new_game_data


def calculate_rolling_averages(team, all_data, num_games=5):
    team_data = all_data[all_data['Team'] == team].sort_values(by='Date')
    
    # Calculate rolling averages for the last 'num_games'
    rolling_avg = team_data[['Off_Pts', 'Def_Pts', 'Time_of_Possession_Seconds', '3DConv', '4DConv']].rolling(window=num_games).mean().iloc[-1]
    
    return {
        'Avg_Off_Pts_Last_5': rolling_avg['Off_Pts'],
        'Avg_Def_Pts_Last_5': rolling_avg['Def_Pts'],
        'Avg_Time_of_Possession_Seconds_Last_5': rolling_avg['Time_of_Possession_Seconds'],
        'Avg_3DConv_Last_5': rolling_avg['3DConv'],
        'Avg_4DConv_Last_5': rolling_avg['4DConv']
    }
# Convert to DataFrame
input_df = pd.DataFrame([new_game_data])
#X = all_data[['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5', 
#'Avg_Time_of_Possession_Seconds_Last_5', 'Avg_3DConv_Last_5', 'Avg_4DConv_Last_5'] + team_columns + player_columns + day_columns + year_columns]
# Ensure the DataFrame columns are in the correct order
team_rolling_averages = calculate_rolling_averages(new_game_data['Team'], data)
opp_rolling_averages = calculate_rolling_averages(new_game_data['Opponent'], data)

# Add team rolling averages
new_game_data.update(team_rolling_averages)

# Add opponent rolling averages with a different naming convention
new_game_data.update({f"{key}": value for key, value in opp_rolling_averages.items()})

# Apply the other functions
new_game_data = encode_teams(new_game_data)
new_game_data = get_last_game_player_data(new_game_data)
new_game_data = encode_day_and_year(new_game_data)

# Convert to DataFrame
input_df = pd.DataFrame([new_game_data])
#exp = pd.DataFrame([data])
# X = all_data[['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5', 
#                   'Avg_Time_of_Possession_Seconds_Last_5', 'Avg_3DConv_Last_5', 'Avg_4DConv_Last_5'] + team_columns + player_columns + day_columns + year_columns]
# Ensure the DataFrame columns are in the correct order
expected_columns = ['Spread', 'Total', 'Home', 'Avg_Off_Pts_Last_5', 'Avg_Def_Pts_Last_5', 
                    'Avg_Time_of_Possession_Seconds_Last_5', 'Avg_3DConv_Last_5', 'Avg_4DConv_Last_5'] + team_columns + player_columns + day_columns + year_columns

columns_to_drop = ['Opp_Pts', 'Team_Pts', 'Team_Yds', 'Team_Yds_Lost_Sacks']
input_df = input_df.drop(columns=columns_to_drop, errors='ignore')
expected_columns = [col for col in expected_columns if col not in columns_to_drop]

#input_df[expected_columns].to_csv("SD.csv", index=False)
X_new = input_df[expected_columns]

#X_new.to_csv("SDF.csv", index=False)

if len(expected_columns) != input_size:
    unavailable_columns = ['3DConv', '4DConv', 'Time_of_Possession', 'Time_of_Possession_Seconds', 
                       'Off_Pts', 'Def_Pts', 'Cmp', 'Att', 'TD', 'Int', 'Sk', 'Y/A', 'NY/A', 
                       'Cmp%', 'Rate', 'Rush_Yds', 'Pnt', 'Punt_Yds']

    # Drop these columns from the input dataframe (ignore errors if they aren't present)
    input_df = input_df.drop(columns=unavailable_columns, errors='ignore')

    # Now check again against the test dataset
    test_columns = data.columns
    input_columns = input_df.columns

    extra_columns = [col for col in input_columns if col not in test_columns]
    missing_columns = [col for col in test_columns if col not in input_columns]
    print(f"Extra columns: {extra_columns}")
    print(f"Missing columns: {missing_columns}")
    raise ValueError(f"Model expects {input_size} features, but got {len(expected_columns)}")

# Convert DataFrame to a PyTorch tensor
X_tensor = torch.tensor(X_new.values, dtype=torch.float32)

# Ensure the tensor matches the expected shape
if X_tensor.shape[1] != input_size:
    raise ValueError(f"Expected input size of {input_size}, but got {X_tensor.shape[1]}")

# Pass the tensor through the model
with torch.no_grad():  # Disable gradient calculations since we're in evaluation mode
    model_output = model(X_tensor)

# Print the model output (this could be probabilities, a score, or anything based on how your model works)
print(f"Model output: {model_output}")

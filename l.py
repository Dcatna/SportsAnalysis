import pandas as pd

player_data = pd.read_csv("new_player_data.csv")
game_data = pd.read_csv("nfl_merged_corrected.csv")

player_data.rename(columns={'Opp': 'Opponent'}, inplace=True)
game_data.rename(columns={'Opp': 'Opponent'}, inplace=True)

# Perform the merge on Date, Team, and Opponent
merged_data = pd.merge(
    player_data,
    game_data,
    on=['Date', 'Team', 'Opponent'],
    suffixes=('_player', '_game')
)

# One-Hot Encode the players in the merged dataset
player_one_hot = pd.get_dummies(merged_data['Player'], prefix='player')

# Aggregate the one-hot encoded players by the game (using 'Date', 'Team', 'Opponent')
game_player_one_hot = pd.concat([merged_data[['Date', 'Team', 'Opponent']], player_one_hot], axis=1)
game_player_one_hot = game_player_one_hot.groupby(['Date', 'Team', 'Opponent']).max().reset_index()

# Merge the one-hot encoded players back into the original game data
final_data = pd.merge(game_data, game_player_one_hot, on=['Date', 'Team', 'Opponent'], how='left')
final_data = final_data.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Display the first few rows of the final dataset
final_data_head = final_data.head()
final_data.to_csv("insp1.csv", index=False)


print(final_data_head)





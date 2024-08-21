import pandas as pd

# Load your data
player_df = pd.read_csv('all_player_data.csv')

# Step 1: Remove rows that contain repeated headers or non-data text
player_df = player_df[~player_df['Player'].str.contains('Player|Passing|Rushing|Receiving|Fumbles', na=False)]

# Step 2: Group by unique game identifiers and count occurrences
game_grouped = player_df.groupby(['Season', 'Date', 'Teams']).size()

# Step 3: Identify games with excessive rows
excessive_rows = game_grouped[game_grouped > 30]  # Adjust the threshold if necessary

print(f"Games with excessive player rows:\n{excessive_rows}")

# Step 4: Filter out games with excessively high player counts
normal_games = game_grouped[game_grouped <= 30]  # Example threshold for reasonable row counts

# Subset the dataframe to include only games with a reasonable number of rows
player_df_cleaned = player_df[player_df.set_index(['Season', 'Date', 'Teams']).index.isin(normal_games.index)]

# Step 5: Drop duplicates again as a final cleanup step
player_df_cleaned = player_df_cleaned.drop_duplicates()

# Save cleaned data
player_df_cleaned.to_csv('cleaned_player_data_final.csv', index=False)

print(f"Cleaned data contains {player_df_cleaned.shape[0]} rows.")

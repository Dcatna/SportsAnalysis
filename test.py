import pandas as pd

# Load the two datasets
gamelogs_df = pd.read_csv("nfl_pts_and_vegas_with_gamelogs.csv")
teams_df = pd.read_csv("nfl_pts_and_vegas_with_teams.csv")

# Identify common columns to merge on
# Assuming 'Season', 'Week', 'Team', and 'Opp' are the keys
common_keys = ['Season', 'Week', 'Day', 'Date', 'Win', 'OT', 'Home', 'Off_Pts', 'Def_Pts', 'G', 'Spread', 'Total', 'Point_Diff', 'Spread_Covered', 'Actual_vs_Expected']

# Merge the datasets
merged_df = pd.merge(gamelogs_df, teams_df, on=common_keys, how='left')

# Handle any duplicate columns (if any)
# You may want to drop duplicate columns, e.g., if both datasets have 'Off_Pts', keep only one
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Save the merged dataset to a new CSV file
merged_df.to_csv("nfl_merged_final.csv", index=False)

# Print the first few rows to verify
print(merged_df.head())

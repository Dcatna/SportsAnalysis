import pandas as pd

# Load the first dataset (e.g., team and opponent information with features)
df1 = pd.read_csv('nfl_pts_and_vegas_with_gamelogs.csv')

# Load the second dataset (e.g., gamelogs with detailed stats)
df2 = pd.read_csv('nfl_pts_and_vegas_with_teams.csv')

# Print the first few rows of each dataset to inspect them
print(df1.head())
print(df2.head())


# Check for duplicates in df1
duplicate_keys_df1 = df1[df1.duplicated(subset=['Season', 'Week', 'Day', 'Date', 'Home', 'Off_Pts', 'Def_Pts'], keep=False)]
print(f"Duplicates in df1: {duplicate_keys_df1}")

# Check for duplicates in df2
duplicate_keys_df2 = df2[df2.duplicated(subset=['Season', 'Week', 'Day', 'Date', 'Home', 'Off_Pts', 'Def_Pts'], keep=False)]
print(f"Duplicates in df2: {duplicate_keys_df2}")


# Identify duplicates in the first DataFrame
df1_duplicates = df1[df1.duplicated(subset=['Season', 'Week', 'Day', 'Date', 'Home', 'Off_Pts', 'Def_Pts'], keep=False)]
print(f"Number of duplicates in df1: {df1_duplicates.shape[0]}")
print(df1_duplicates)

# Identify duplicates in the second DataFrame
df2_duplicates = df2[df2.duplicated(subset=['Season', 'Week', 'Day', 'Date', 'Home', 'Off_Pts', 'Def_Pts'], keep=False)]
print(f"Number of duplicates in df2: {df2_duplicates.shape[0]}")
print(df2_duplicates)

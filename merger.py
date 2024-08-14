import pandas as pd

# Load the game logs dataset
game_logs = pd.read_csv("nfl_gamelogs_2014-2023.csv")

# Clean the data by selecting relevant columns and ensuring consistent naming
game_logs = game_logs[['Season', 'Week', 'Team', 'Opp', 'Tm', 'Opp.1', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Sk', 'Yds.1', 'Y/A', 'NY/A', 'Cmp%', 'Rate', 'Att.1', 'Yds.2', 'Y/A.1', 'TD.1', 'Pnt', 'Yds.3', '3DConv', '3DAtt', '4DConv', '4DAtt', 'ToP']]

# Rename columns to meaningful names (Optional)
game_logs.rename(columns={
    'Tm': 'Team_Pts',
    'Opp.1': 'Opp_Pts',
    'Yds': 'Team_Yds',
    'Yds.1': 'Team_Yds_Lost_Sacks',
    'Yds.2': 'Rush_Yds',
    'Yds.3': 'Punt_Yds',
    'ToP': 'Time_of_Possession'
}, inplace=True)

# Map full team names to abbreviations if necessary
team_abbr = {
    'Arizona Cardinals': 'CRD',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'RAV',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GNB',
    'Houston Texans': 'HTX',
    'Indianapolis Colts': 'CLT',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KAN',
    'Los Angeles Chargers': 'SDG',
    'Los Angeles Rams': 'RAM',
    'Las Vegas Raiders': 'RAI',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NWE',
    'New Orleans Saints': 'NOR',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SFO',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TAM',
    'Tennessee Titans': 'OTI',
    'Washington Commanders': 'WAS'
    # Add more as necessary
}

# Apply team abbreviation mapping if needed
game_logs['Team'] = game_logs['Team'].map(team_abbr).fillna(game_logs['Team'])
game_logs['Opp'] = game_logs['Opp'].map(team_abbr).fillna(game_logs['Opp'])

# Load your existing merged dataset
merged_dataset = pd.read_csv("nfl_pts_and_vegas_with_features.csv")

# Merge the datasets on Season, Week, Team, and Opp
merged_with_gamelogs = pd.merge(
    merged_dataset,
    game_logs,
    on=['Season', 'Week', 'Team', 'Opp'],
    how='left'
)

# Check for NaNs in the merge and fill them or drop if necessary
merged_with_gamelogs.fillna(0, inplace=True)  # Or drop with .dropna()

# Save the new merged dataset
merged_with_gamelogs.to_csv("nfl_pts_and_vegas_with_gamelogs.csv", index=False)

# Print the first few rows to verify
print(merged_with_gamelogs.head())

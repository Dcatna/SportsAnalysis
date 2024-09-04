import pandas as pd
from datetime import datetime

# Load the datasets
game_data = pd.read_csv("nfl_merged_corrected.csv")
player_data = pd.read_csv("all_player_data_cleaned.csv")

# Function to convert descriptive date format to YYYY-MM-DD
def convert_date(row):
    return datetime.strptime(f"{row['Date']} {row['Season']}", "%B %d %Y").strftime("%Y-%m-%d")

# Apply the conversion to the game data
game_data['Date'] = game_data.apply(convert_date, axis=1)

# Complete team mapping between the game data and player data
team_mapping = {
    'CRD': 'ARI',  # Arizona Cardinals
    'SDG': 'LAC',  # San Diego Chargers -> Los Angeles Chargers
    'RAM': 'LAR',  # St. Louis Rams -> Los Angeles Rams
    'RAI': 'LVR',  # Oakland Raiders -> Las Vegas Raiders
    'OTI': 'TEN',  # Tennessee Titans
    'HTX': 'HOU',  # Houston Texans
    'CLT': 'IND',  # Indianapolis Colts
    'KAN': 'KC',   # Kansas City Chiefs
    'NOR': 'NO',   # New Orleans Saints
    'NWE': 'NE',   # New England Patriots
    'TAM': 'TB',   # Tampa Bay Buccaneers
    'SFO': 'SF',   # San Francisco 49ers
    'GNB': 'GB',   # Green Bay Packers
    'MIA': 'MIA',  # Miami Dolphins (already matches)
    'BUF': 'BUF',  # Buffalo Bills (already matches)
    'NYJ': 'NYJ',  # New York Jets (already matches)
    'NYG': 'NYG',  # New York Giants (already matches)
    'DEN': 'DEN',  # Denver Broncos (already matches)
    'PIT': 'PIT',  # Pittsburgh Steelers (already matches)
    'CLE': 'CLE',  # Cleveland Browns (already matches)
    'BAL': 'BAL',  # Baltimore Ravens (already matches)
    'CIN': 'CIN',  # Cincinnati Bengals (already matches)
    'CHI': 'CHI',  # Chicago Bears (already matches)
    'DET': 'DET',  # Detroit Lions (already matches)
    'ATL': 'ATL',  # Atlanta Falcons (already matches)
    'CAR': 'CAR',  # Carolina Panthers (already matches)
    'JAX': 'JAX',  # Jacksonville Jaguars (already matches)
    'DAL': 'DAL',  # Dallas Cowboys (already matches)
    'PHI': 'PHI',  # Philadelphia Eagles (already matches)
    'SEA': 'SEA',  # Seattle Seahawks (already matches)
    'MIN': 'MIN',  # Minnesota Vikings (already matches)
    'WAS': 'WAS'   # Washington Football Team (already matches)
}

# Apply the mapping to the game data
game_data['Team'] = game_data['Team'].map(team_mapping)

# Aggregate player data by date and team, summarizing key statistics
aggregated_player_data = player_data.groupby(['Date', 'Tm']).agg({
    'Cmp': 'sum',       # Sum of completions
    'Att': 'sum',       # Sum of attempts
    'Yds': 'sum',       # Sum of yards
    'TD': 'sum',        # Sum of touchdowns
    'Int': 'sum',       # Sum of interceptions
    'Sk': 'sum',        # Sum of sacks
    'Yds.1': 'sum',     # Sum of sack yards
    'Lng': 'mean',      # Average of longest play
    'Tgt': 'sum',       # Sum of targets
    'Rec': 'sum',       # Sum of receptions
    'Yds.3': 'sum',     # Sum of receiving yards
    'TD.2': 'sum',      # Sum of receiving touchdowns
    'Lng.2': 'mean',    # Average of longest reception
    'Fmb': 'sum',       # Sum of fumbles
    'FL': 'sum'         # Sum of fumbles lost
}).reset_index()

# Merge the aggregated player data with the game data
merged_data_fixed = pd.merge(game_data, aggregated_player_data, how='left', left_on=['Date', 'Team'], right_on=['Date', 'Tm'])

# Save the fixed merged dataset to a CSV file
merged_data_fixed.to_csv("nfl_combined_fixed.csv", index=False)

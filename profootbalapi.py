import requests
import pandas as pd

# Function to fetch player stats for a specific team and season by week
def fetch_team_player_stats_by_week(team, season, week):
    api_key = "your_api_key_here"  # Replace with your actual API key
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerGameStatsByTeam/{season}/{week}/{team}"
    headers = {'Ocp-Apim-Subscription-Key': '82b0838ed879476c8ce8292e22ab58e2'}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    print(data)
    return data

# Function to determine if a player is impactful based on their position and stats
def is_impactful(player):
    if isinstance(player, dict):  # Ensure the player is a dictionary
        position = player.get('Position')
    
        if position in ['QB', 'RB', 'WR', 'TE']:
            # Offensive players
            yards = player.get('PassingYards', 0) + player.get('RushingYards', 0) + player.get('ReceivingYards', 0)
            touchdowns = player.get('PassingTouchdowns', 0) + player.get('RushingTouchdowns', 0) + player.get('ReceivingTouchdowns', 0)
            return yards > 50 or touchdowns > 1  # Adjust thresholds for per-game stats
        elif position in ['DL', 'LB', 'DB']:
            # Defensive players
            tackles = player.get('SoloTackles', 0) + player.get('AssistedTackles', 0)
            sacks = player.get('Sacks', 0)
            interceptions = player.get('Interceptions', 0)
            return tackles > 2 or sacks > 1 or interceptions > 0  # Adjust thresholds for per-game stats
    return False

# Function to add player stats as features to the game dataset
def add_player_features(game_data, impactful_players, week, team):
    # Dictionary to hold new columns
    new_columns = {}

    for player in impactful_players:
        # Collect stats to add as new columns for the specific game
        new_columns[f'{player["Name"]}_Yards'] = game_data.apply(
            lambda row: player.get('PassingYards', 0) + player.get('RushingYards', 0) + player.get('ReceivingYards', 0)
            if row['Week'] == week and row['Team'] == team else 0, axis=1)
        new_columns[f'{player["Name"]}_Touchdowns'] = game_data.apply(
            lambda row: player.get('PassingTouchdowns', 0) + player.get('RushingTouchdowns', 0) + player.get('ReceivingTouchdowns', 0)
            if row['Week'] == week and row['Team'] == team else 0, axis=1)

    # Add all new columns at once using pd.concat
    game_data = pd.concat([game_data, pd.DataFrame(new_columns)], axis=1)
    
    return game_data

# Main function to fetch player data for all teams, identify impactful players, and add their stats to the game data
def main():
    teams = ["crd", "atl", "rav", "buf", "car", "chi", "cin", "cle", "dal", "den", "det", "gnb", "htx", "clt", "jax", "kan", 
             "sdg", "ram", "rai", "mia", "min", "nwe", "nor", "nyg", "nyj", "phi", "pit", "sea", "sfo", "tam", "oti", "was"]
    season = 2023
    
    # Load your game data from a CSV file
    game_data = pd.read_csv('nfl_merged_corrected.csv')  # Path to your actual game data file
    n=0
    for team in teams:
        for week in game_data['Week'].unique():
            print(week, season, team)
            player_data = fetch_team_player_stats_by_week(team, season, week)
            impactful_players = [player for player in player_data if is_impactful(player)]
            print(impactful_players)
            game_data = add_player_features(game_data, impactful_players, week, team)
            n+=1
    
    # Output the updated game data (for verification)
    print(game_data.head())

    # Save the updated game data to a new CSV file
    game_data.to_csv('nfl_merged_corrected_with_player_stats.csv', index=False)

if __name__ == "__main__":
    main()

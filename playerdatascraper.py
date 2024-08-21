import numpy as np
import pandas as pd
import random
import time
import os

# Load your existing data
df = pd.read_csv('nfl_gamelogs_2014-2023.csv')

# Extract month and day
df['Month'] = df['Date'].apply(lambda x: x.split()[0])  # Extracts the month name
df['Day'] = df['Date'].apply(lambda x: x.split()[1])  # Extracts the day number

# Convert month names to numerical values and format with leading zeros
df['Month'] = pd.to_datetime(df['Month'], format='%B').dt.month.apply(lambda x: f"{x:02d}")
df['Day'] = df['Day'].apply(lambda x: f"{int(x):02d}")

# Team dictionary
team_dict = {
    'Arizona Cardinals': 'crd', 
    'Atlanta Falcons': 'atl', 
    'Baltimore Ravens': 'rav', 
    'Buffalo Bills': 'buf', 
    'Carolina Panthers': 'car', 
    'Chicago Bears': 'chi',
    'Cincinnati Bengals': 'cin', 
    'Cleveland Browns': 'cle', 
    'Dallas Cowboys': 'dal', 
    'Denver Broncos': 'den', 
    'Detroit Lions': 'det', 
    'Green Bay Packers': 'gnb', 
    'Houston Texans': 'htx', 
    'Indianapolis Colts': 'clt', 
    'Jacksonville Jaguars': 'jax', 
    'Kansas City Chiefs': 'kan', 
    'Los Angeles Chargers': 'sdg', 
    'Los Angeles Rams': 'ram', 
    'Las Vegas Raiders': 'rai', 
    'Miami Dolphins': 'mia', 
    'Minnesota Vikings': 'min', 
    'New England Patriots': 'nwe', 
    'New Orleans Saints': 'nor', 
    'New York Giants': 'nyg', 
    'New York Jets': 'nyj', 
    'Oakland Raiders': 'rai',
    'Philadelphia Eagles': 'phi', 
    'Pittsburgh Steelers': 'pit', 
    'San Diego Chargers': 'sdg',
    'San Francisco 49ers': 'sfo',
    'St. Louis Rams': 'ram', 
    'Seattle Seahawks': 'sea', 
    'Tampa Bay Buccaneers': 'tam', 
    'Tennessee Titans': 'oti', 
    'Washington Commanders': 'was', 
    'Washington Redskins': 'was', 
    'Washington Football Team': 'was'
}

# Function to create the URL and teams string
def create_url_and_teams(row):
    base_url = f"https://www.pro-football-reference.com/boxscores/{row['Season']}{row['Month']}{row['Day']}0"
    
    if row['Unnamed: 6'] == '@':
        team_in_url = team_dict.get(row['Opp'], row['Opp']).lower()
    else:
        team_in_url = team_dict.get(row['Team'], row['Team']).lower()
    
    url = f"{base_url}{team_in_url}.htm"
    teams = f"{team_dict.get(row['Team'], row['Team'])}_vs_{team_dict.get(row['Opp'], row['Opp'])}"
    
    return pd.Series([url, teams])

# Apply the function to each row
df[['URL', 'Teams']] = df.apply(create_url_and_teams, axis=1)

# Scrape and clean data
all_player_data = []

for index, row in df.iterrows():  # Test with first 5 rows
    url = row['URL']
    teams = row['Teams']

    try:
        print(f"Scraping: {url}")
        off_df = pd.read_html(url, header=1, attrs={'id': 'player_offense'})[0]
        
        # Drop any rows that are headers duplicated as data or where 'Player' is NaN
        off_df = off_df[off_df['Player'] != 'Player']
        off_df = off_df.dropna(subset=['Player'])
        
        # Add additional context columns
        off_df['Season'] = row['Season']
        off_df['Date'] = f"{row['Season']}-{row['Month']}-{row['Day']}"
        off_df['Teams'] = teams

        # Append each row of the table to the list
        all_player_data.append(off_df)
        print(f"Scraped {len(off_df)} rows from {url}")
        time.sleep(random.randint(8, 10))

    except Exception as e:
        print(f"Error scraping {url}: {e}")

# Combine all data into a single DataFrame
all_player_data_df = pd.concat(all_player_data, ignore_index=True)

# Remove any remaining duplicate rows
all_player_data_df.drop_duplicates(inplace=True)

# Save the final data to a CSV
all_player_data_df.to_csv('all_player_data_cleaned.csv', index=False)

print(f"Scraping completed. Final dataset has {all_player_data_df.shape[0]} rows.")

import numpy as np
import pandas as pd
import random
import time
import os

seasons = [str(season) for season in range(2014, 2024)]
print(f"Number of seasons: {len(seasons)}")

teams = ["crd", "atl", "rav", "buf", "car", "chi", "cin", "cle", "dal", "den", "det", "gnb", "htx", "clt", "jax", "kan", 
         "sdg", "ram", "rai", "mia", "min", "nwe", "nor", "nyg", "nyj", "phi", "pit", "sea", "sfo", "tam", "oti", "was"]

if not os.path.exists("C:/Users/ddcat/OneDrive/Desktop/Projects/SportsAnalysis/nfl_gamelogs_2014-2023.csv"):
    #
    # GETTING GAME DATA
    #
    
    nfl_df = pd.DataFrame()
    for season in seasons:
        for team in teams:
            url = "https://www.pro-football-reference.com/teams/" + team + "/" + season + "/gamelog/"
            print(url)

            off_df = pd.read_html(url, header=1, attrs={'id': 'gamelog' + season})[0]
            def_df = pd.read_html(url, header=1, attrs={'id': 'gamelog_opp' + season})[0]

            team_df = pd.concat([off_df, def_df], axis=1)
            team_df.insert(loc=0, column="Season", value=season)
            team_df.insert(loc=2, column="Team", value=team.upper())

            nfl_df = pd.concat([nfl_df, team_df], ignore_index=True)

            time.sleep(random.randint(8, 10))

    print(nfl_df)

    nfl_df.to_csv("nfl_gamelogs_2014-2023.csv", index=False)
    print(nfl_df.shape)
else:
    nfl_df = pd.read_csv("C:/Users/ddcat/OneDrive/Desktop/Projects/SportsAnalysis/nfl_gamelogs_2014-2023.csv")  # Adjust the path as necessary

    #print(nfl_df)   
    nfl_pts_df = nfl_df.drop(nfl_df.columns[12:], axis=1)
    nfl_pts_df = nfl_pts_df.drop(nfl_pts_df.columns[5:6], axis=1)

    column_names = {'Unnamed: 4' : 'Win', 'Unnamed: 6' : 'Home', 'Tm' : 'Off_Pts', 'Opp.1' : 'Def_Pts'}
    nfl_pts_df = nfl_pts_df.rename(columns=column_names)
    print(nfl_pts_df.info(verbose=True))

    # Map 'Opp' to three-letter abbreviations
    team_dict = {
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
        'Oakland Raiders': 'RAI',
        'Miami Dolphins': 'MIA', 
        'Minnesota Vikings': 'MIN', 
        'New England Patriots': 'NWE', 
        'New Orleans Saints': 'NOR', 
        'New York Giants': 'NYG', 
        'New York Jets': 'NYJ', 
        'Philadelphia Eagles': 'PHI', 
        'Pittsburgh Steelers': 'PIT', 
        'St. Louis Rams': 'RAM',
        'San Diego Chargers': 'SDG',
        'San Francisco 49ers': 'SFO', 
        'Seattle Seahawks': 'SEA', 
        'Tampa Bay Buccaneers': 'TAM', 
        'Tennessee Titans': 'OTI', 
        'Washington Commanders': 'WAS', 
        'Washington Redskins': 'WAS', 
        'Washington Football Team': 'WAS'
    }

    nfl_pts_df = nfl_pts_df.replace({'Opp': team_dict})

    # Convert 'Win' column to 1 = Win or 0 = Loss
    nfl_pts_df['Win'] = nfl_pts_df['Win'].apply(lambda x: 1 if x == 'W' else 0)

    # Convert 'OT' column to 1 = OT or 0 = No OT
    nfl_pts_df['OT'] = nfl_pts_df['OT'].apply(lambda x: 1 if x == 'OT' else 0)

    nfl_pts_df['Home'] = nfl_pts_df['Home'].apply(lambda x: 0 if x == '@' else 1)
    print(nfl_pts_df)

    #
    # GET VEGAS LINE DATA
    #
    if not os.path.exists("C:/Users/ddcat/OneDrive/Desktop/Projects/SportsAnalysis/nfl_vegas_lines_2014-2023.csv"):

        veg_df = pd.DataFrame()
        for season in seasons:
            for team in teams:
                url = "https://www.pro-football-reference.com/teams/" + team + "/" + season + "_lines.htm"
                print(url)

                lines_df =  pd.read_html(url, header=0, attrs={'id': 'vegas_lines'})[0]

                lines_df.insert(loc=0, column='Season', value=season)
                lines_df.insert(loc=2, column='Team', value=team.upper())

                veg_df = pd.concat([veg_df, lines_df], ignore_index=True)
                time.sleep(random.randint(8, 10))

        veg_df.to_csv('nfl_vegas_lines_2014-2023.csv', index=False)
        
    else:
        veg_df = pd.read_csv("C:/Users/ddcat/OneDrive/Desktop/Projects/SportsAnalysis/nfl_vegas_lines_2014-2023.csv")
        veg_df = veg_df.drop(veg_df.columns[6:], axis=1)
        column_names = {"G#":"G", "Over/Under":"Total"}
        veg_df = veg_df.rename(columns=column_names)
        print(veg_df.info(verbose=True))

        veg_df = veg_df.query('(Season <= 2020 and G < 17) or (Season >= 2021 and G < 18)')
        print(veg_df.shape)
        #print(nfl_pts_df.shape)

        veg_df['Home'] = veg_df['Opp'].apply(lambda x: 0 if x[0] == '@' else 1)
        veg_df['Opp'] = veg_df['Opp'].apply(lambda x: x[1:] if x[0] == '@' else x)

        abbr_dict = {
        'OAK': 'RAI', 
        'LVR': 'RAI', 
        'STL': 'RAM', 
        'LAR': 'RAM', 
        'LAC': 'SDG', 
        'IND': 'CLT', 
        'HOU': 'HTX', 
        'BAL': 'RAV', 
        'ARI': 'CRD', 
        'TEN': 'OTI'
        }

        veg_df = veg_df.replace({'Opp': abbr_dict})
        print(veg_df.shape)
        print(nfl_pts_df.shape)
        # nfl_pts_df['Opp'] = nfl_pts_df['Opp'].astype(str)
        # veg_df['Opp'] = veg_df['Opp'].astype(str)
        merged_df = pd.merge(nfl_pts_df, veg_df, on=['Season', 'Team', 'Opp', 'Home'])
        print(nfl_pts_df.query('Season == 2014 and Team == "CRD"'))
        print(veg_df.query('Season == 2014 and Team == "CRD"'))
        print(merged_df.query('Season == 2014 and Team == "CRD"'))
        
        merged_df.to_csv('nfl_pts_and_vegas_2014-2023.csv', index=False)


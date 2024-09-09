import pandas as pd

all_data = pd.read_csv("all_data.csv")


def remove_duplicate_games(df):
    # Create a new column that combines Season, Date, Team, and Opponent
    df['game_id'] = df.apply(lambda row: tuple(sorted([row['Team'], row['Opponent']])) + (row['Season'], row['Date']), axis=1)
    
    # Drop the duplicates based on the game_id
    df = df.drop_duplicates(subset='game_id')
    
    # Drop the game_id column as it's no longer needed
    df = df.drop(columns='game_id')
    
    return df
all_data = remove_duplicate_games(all_data)
all_data = pd.get_dummies(all_data, columns=['Day'], prefix='Day', drop_first=False)
day_columns = [col for col in all_data.columns if col.startswith('Day_')]
all_data[day_columns] = all_data[day_columns].astype(int)

all_data = pd.get_dummies(all_data, columns=['Season'], prefix='Year', drop_first=False)

year_columns = [col for col in all_data.columns if col.startswith("Year_")]
all_data[year_columns] = all_data[year_columns].astype(int)
# Now concatenate the one-hot encoded years back to the original dataframe
#all_data = pd.concat([all_data, all_data], axis=1)

# Optionally, you can drop the original 'Season' column if you don't need it anymore
#all_data = all_data.drop('Season', axis=1)
# Apply the function
#all_data_cleane = remove_duplicate_games(all_data)
all_data.to_csv("t.csv", index=False)
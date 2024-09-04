import pandas as pd

# Load your data
player_data= pd.read_csv('all_player_data_cleaned.csv')
player_data['Date'] = pd.to_datetime(player_data['Date'])

# Format the 'Date' column to "Month Day" (e.g., "September 8")
player_data['Date'] = player_data['Date'].dt.strftime('%B %#d')

# Display the modified DataFrame to check the results
print(player_data[['Date']])
player_data[['Team', 'Opp']] = player_data['Teams'].str.split('_vs_', expand=True)

# Convert the extracted team names to uppercase to match your desired format
player_data['Team'] = player_data['Team'].str.upper()
player_data['Opp'] = player_data['Opp'].str.upper()

# Display the modified DataFrame to check the results
print(player_data[['Teams', 'Team', 'Opp']])
print(player_data)
player_data.to_csv("new_player_data.csv", index=False)
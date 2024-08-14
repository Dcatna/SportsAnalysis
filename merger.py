import pandas as pd

# Load the dataset that includes 'Team' and 'Opp' columns
merged_dataset = pd.read_csv("nfl_pts_and_vegas_with_features.csv")

# Perform one-hot encoding on 'Team' and 'Opp' columns
team_encoded = pd.get_dummies(merged_dataset[['Team', 'Opp']], prefix=['Team', 'Opp'])

# Combine the one-hot encoded columns with the original dataset
merged_dataset_with_teams = pd.concat([merged_dataset, team_encoded], axis=1)

# Drop the original 'Team' and 'Opp' columns if they are no longer needed
merged_dataset_with_teams = merged_dataset_with_teams.drop(columns=['Team', 'Opp'])

# Convert only the boolean columns (True/False) to integers (1/0)
# Here we identify the boolean columns and apply the conversion
boolean_columns = merged_dataset_with_teams.select_dtypes(include='bool').columns
merged_dataset_with_teams[boolean_columns] = merged_dataset_with_teams[boolean_columns].astype(int)

# Save the updated dataset with one-hot encoded team features
merged_dataset_with_teams.to_csv("nfl_pts_and_vegas_with_teams.csv", index=False)

# Print the first few rows to verify the changes
print(merged_dataset_with_teams.head())

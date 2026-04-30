import pandas as pd

matches = pd.read_csv("data/raw/results.csv")
rankings = pd.read_csv("data/raw/FIFA_Soccer_Rankings.csv")

print("=== Matches ===")
print(matches.head())

print("\n=== Rankings ===")
print(rankings.head())


#how many rows in each table
print("match rows:", matches.shape[0])
print("ranking rows:", rankings.shape[0])

#what tournament exists in the matches data
print("\nTournament types:")
print(matches["tournament"].value_counts())

#are there any missing values in the matches data?
print("\nMissing values in matches:")
print(matches.isnull().sum())

#cleaning

#step 1 - drop rows with missing scores
matches = matches.dropna(subset=["home_score", "away_score"])
print("After dropping missing scores:", matches.shape[0], "matches")

#step 2 - remove friendlies
matches = matches[matches["tournament"]!= "Friendly"]
print("After removing friendlies:", matches.shape[0], "matches")

#step 3 - keep only the matches from 1993 and onwards
matches["date"] = pd.to_datetime(matches["date"])
matches = matches[matches["date"] >= "1993-01-01"]
print("After filtering to 1993+:", matches.shape[0], "matches")

#step 4 - result label creation
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 0
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 2
    
matches["result"] = matches.apply(get_result, axis=1)
print("\nResult value coutns:")
print(matches["result"].value_counts())

matches.to_csv("data/processed/matches_clean.csv", index=False)
print("\nSaved cleaned data to data/processed/matches_clean.csv")
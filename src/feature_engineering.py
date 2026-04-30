import pandas as pd

# load all the data
matches = pd.read_csv("data/processed/matches_clean.csv")
rankings = pd.read_csv("data/raw/FIFA_Soccer_Rankings.csv")

# convert dates to same formatting
matches["date"] = pd.to_datetime(matches["date"])
rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])

# keep only the required columns
rankings = rankings[["rank_date", "country_full", "rank"]]

# correct the team/country name mismatches between the two data-sets
name_fixes = {
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "Iran": "IR Iran",
    "North Macedonia": "FYR Macedonia",
    "United States": "USA",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Eswatini": "Swaziland",
    "Cape Verde": "Cape Verde Islands",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Brunei": "Brunei Darussalam",
    "Macedonia": "FYR Macedonia",
}
matches["home_team"] = matches["home_team"].replace(name_fixes)
matches["away_team"] = matches["away_team"].replace(name_fixes)

# Drop matches involving teams not in FIFA rankings
all_ranked_teams = set(rankings["country_full"].unique())
matches = matches[
    matches["home_team"].isin(all_ranked_teams) &
    matches["away_team"].isin(all_ranked_teams)
]
print("Matches after dropping non-FIFA teams:", matches.shape[0])

# Sort both tables by date — required for merge_asof
matches = matches.sort_values("date")
rankings = rankings.sort_values("rank_date")

# join home team rankings
matches = pd.merge_asof(
    matches,
    rankings.rename(columns={"country_full": "home_team", "rank": "home_rank"}),
    left_on="date",
    right_on="rank_date",
    by="home_team"
)

# join away team rankings
matches = pd.merge_asof(
    matches,
    rankings.rename(columns={"country_full": "away_team", "rank": "away_rank"}),
    left_on="date",
    right_on="rank_date",
    by="away_team"
)

# Calculate rank differences
matches["rank_difference"] = matches["home_rank"] - matches["away_rank"]

print("\nRank difference added:")
print(matches[["home_team", "away_team", "home_rank", "away_rank", "rank_difference"]].head(10))
print("\nMissing home ranks:", matches["home_rank"].isnull().sum())
print("Missing away ranks:", matches["away_rank"].isnull().sum())

# drop matches where a ranking for either team could not be found
matches = matches.dropna(subset=["home_rank", "away_rank"])
print("Matches after dropping missing ranks:", matches.shape[0])

# Feature #2 Recent form
# For each team what % of their last 5 games did they win?
# Checks for a teams winning momentum

def get_recent_form(team, date, matches_df, n=5):
    # get all past matches fr this team, both home and away
    past = matches_df[
        ((matches_df["home_team"] == team) |
        (matches_df["away_team"] == team)) & 
        (matches_df["date"] < date)
    ].tail(n) # gets the last n rows

    if len(past) == 0:
        return 0.5 # if there is no history assume average form
    
    wins = 0
    for _, row in past.iterrows():
        if row["home_team"] == team and row ["result"] == 0:
            wins += 1 # home win
        elif row["away_team"] == team and row["result"] == 2:
            wins += 1 # away win

    return wins/ len(past)

# apply this to all matches
print("Calculating home team form...")
matches["home_form"] = matches.apply(
    lambda row: get_recent_form(row["home_team"], row["date"], matches), axis=1
)

print("Calculating away team form...")
matches["away_form"] = matches.apply(
    lambda row: get_recent_form(row["away_team"], row["date"], matches), axis=1
)

print("-- Form features added --")
print(matches[["home_team", "away_team", "home_form", "away_form"]].head(10))


# saving fully featured dataset 
matches.to_csv("data/processed/matches_featured.csv", index=False)
print("Saved! Shape:", matches.shape)
print("\nFinal columns:", list(matches.columns))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# load featured dataset built using feature_engineering.py
matches = pd.read_csv("data/processed/matches_featured.csv")

# columns that the model will learn from
# want to exclude team names, scores, dates since the model cannot use those
feature_columns = [
    "home_rank",
    "away_rank",
    "rank_difference",
    "home_form",
    "away_form",
    "neutral"
]

# X is the features (inputs), y is the result (what is to be predicted by the model)
X = matches[feature_columns]
y = matches["result"]

print("Training data shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())

# split data into training set and test set
# test_size = 0.2 , indicates the model uses 20% of data for testing and 80% of data for training
# random_state = 42 fixes randomness so the split is the same each time
# stratify=y makes sure the 49/29/22 class split is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# create the model
# class_weight = 'balanceed' compensates for the home win imbalance within the dataset
# n_estimators = 100 --> means the forest has 100 decision trees
# random_state = 42 --> makes results reproducible

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# training the model 
# Providing it the inputs (X_train) and correct answers (y_train)
model.fit(X_train, y_train)
print("\n Model training complete! ")

# test the model on data it has never seen before
y_pred = model.predict(X_test)

# measure the model's prediction accuracy
print("\n Accuracy:", round(accuracy_score(y_test, y_pred),3))
print("\n Detailed Breakdown:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# save trained model to a file
# allows for streamlit to load it isntantaneously without retraining
joblib.dump(model, "models/best_model.joblib")
print("\nModel saved to models/best_model.joblib")

# save feature column names
# need these for Streamlit to make sure inputs are in the correct order
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

print("Feature columns saved to models/feature_columns.json")


# XGBoost Model Training (different model type)

xgb_model = XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 4,
    use_label_encoder = False,
    eval_metric = "mlogloss",
    random_state = 42
)

xgb_model.fit(X_train, y_train)
print("\nXGBoost model trained!")

xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = round(accuracy_score(y_test, xgb_pred), 3)
rf_accuracy = round(accuracy_score(y_test, y_pred), 3)

print("\n -- Model Comparison --")
print(f"Random Forest model accuracy: {rf_accuracy}")
print(f"XGBoost model accuracy: {xgb_accuracy}")


print("\nXGBoost detailed breakdown:")
print(classification_report(y_test, xgb_pred, target_names=["Home Win", "Draw", "Away Win"]))

# save whichever model performed better
if xgb_accuracy > rf_accuracy:
    joblib.dump(xgb_model, "models/best_model.joblib")
    print("XGBoost was better — saved as best_model.joblib")
else:
    joblib.dump(model, "models/best_model.joblib")
    print("Random Forest was better — saved as best_model.joblib")
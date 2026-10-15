# FIFA WORLD CUP 2026 - LINEAR REGRESSION 2.2
# Predicting goals scored by a team in a single match

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
file_name = "FIFA_World_Cup_2026_LR2_2_Data_Simplified.xlsx"
data_raw = pd.read_excel(file_name, sheet_name="LR2_2_Data")

# EXPLANATORY VARIABLES AND TARGET
explanatory = ["Team_Elo_Pre", "Opp_Elo_Pre", "Team_GF_L10", "Opp_GA_L10", "Team_WinRate_L10", "Rest_Days", "Home_Nation", "Knockout"]
target = "GoalsScored"

# CHECK DATA
print("Number of rows:", data_raw.shape[0])
print("Number of columns:", data_raw.shape[1])
print("Unique matches:", data_raw["MatchID"].nunique())
print("Number of explanatory variables:", len(explanatory))
print("Missing values:")
print(data_raw[explanatory + [target]].isna().sum())
print("Duplicate team-match rows:", data_raw.duplicated(subset=["MatchID", "Team"]).sum())

# DATA CLEANING
data = data_raw.dropna(subset=explanatory + [target])
print("Rows after cleaning:", len(data))

# DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")
print(data[explanatory + [target]].describe().round(2))

# DISTRIBUTION OF GOALS SCORED
plt.hist(data["GoalsScored"], bins=8)
plt.title("Distribution of Goals Scored")
plt.xlabel("Goals Scored")
plt.ylabel("Number of Team-Matches")
plt.show()

# RELATIONSHIP BETWEEN TEAM ELO AND GOALS
plt.scatter(data["Team_Elo_Pre"], data["GoalsScored"])
plt.title("Team Elo Rating and Goals Scored")
plt.xlabel("Team Elo Rating Before Match")
plt.ylabel("Goals Scored")
plt.show()

# RELATIONSHIP BETWEEN RECENT SCORING AND GOALS
plt.scatter(data["Team_GF_L10"], data["GoalsScored"])
plt.title("Recent Scoring Form and Goals Scored")
plt.xlabel("Average Goals Scored in Previous 10 Matches")
plt.ylabel("Goals Scored")
plt.show()

# CORRELATION MATRIX
correlation = data[explanatory + [target]].corr()
print("\nCORRELATION MATRIX")
print(correlation.round(2))

sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# SPLIT BY MATCH ID
# Both teams from the same match stay together in either training or testing data
match_ids = data["MatchID"].drop_duplicates()

train_matches, test_matches = train_test_split(match_ids, test_size=0.2, random_state=42)

train_data = data[data["MatchID"].isin(train_matches)]
test_data = data[data["MatchID"].isin(test_matches)]

X_train = train_data[explanatory]
X_test = test_data[explanatory]

y_train = train_data[target]
y_test = test_data[target]

print("\nTraining matches:", train_data["MatchID"].nunique())
print("Testing matches:", test_data["MatchID"].nunique())
print("Training sample size:", len(X_train))
print("Testing sample size:", len(X_test))

# CHECK THAT NO MATCH APPEARS IN BOTH SETS
shared_matches = set(train_data["MatchID"]) & set(test_data["MatchID"])
print("Matches appearing in both sets:", len(shared_matches))

# BASELINE MODEL
baseline_value = y_train.mean()
baseline_predictions = np.ones(len(y_test)) * baseline_value

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_rmse = sqrt(baseline_mse)
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\nBASELINE MODEL")
print("Mean prediction:", round(baseline_value, 2))
print("MAE:", round(baseline_mae, 3))
print("MSE:", round(baseline_mse, 3))
print("RMSE:", round(baseline_rmse, 3))
print("R2:", round(baseline_r2, 3))

# BUILD MULTIPLE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# MAKE PREDICTIONS
predicted_goals = model.predict(X_test)

# MODEL PERFORMANCE
mae = mean_absolute_error(y_test, predicted_goals)
mse = mean_squared_error(y_test, predicted_goals)
rmse = sqrt(mse)
r2 = r2_score(y_test, predicted_goals)

goal_range = np.max(y_test) - np.min(y_test)
nrmse = rmse / goal_range

print("\nLINEAR REGRESSION MODEL")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("Normalised RMSE:", round(nrmse, 3))
print("R2:", round(r2, 3))

# MODEL COEFFICIENTS
coefficients = pd.DataFrame({"Variable": explanatory, "Coefficient": model.coef_})
print("\nMODEL COEFFICIENTS")
print(coefficients.round(4))
print("Intercept:", round(model.intercept_, 4))

# COMPARE WITH BASELINE
print("\nMODEL COMPARISON")
print("Baseline RMSE:", round(baseline_rmse, 3))
print("Regression RMSE:", round(rmse, 3))
print("Baseline MAE:", round(baseline_mae, 3))
print("Regression MAE:", round(mae, 3))

# ACTUAL VS PREDICTED GOALS
plt.scatter(y_test, predicted_goals)
plt.plot([0, np.max(y_test)], [0, np.max(y_test)])
plt.title("Actual vs Predicted Goals")
plt.xlabel("Actual Goals")
plt.ylabel("Predicted Goals")
plt.show()

# FINAL RESULT
print("\nFINAL RESULT")
if rmse < baseline_rmse:
    print("The multiple linear regression model performed better than the baseline model based on RMSE.")
else:
    print("The multiple linear regression model did not perform better than the baseline model based on RMSE.")

print("The R2 score was", round(r2, 3), "which shows how much of the variation in goals was explained by the eight pre-match variables.")

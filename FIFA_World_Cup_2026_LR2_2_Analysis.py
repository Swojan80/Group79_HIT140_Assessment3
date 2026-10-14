# FIFA WORLD CUP 2026 - LINEAR REGRESSION 2.2
# Predicting goals scored by a team in a single match

import os
import json
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold, cross_validate, train_test_split

FILE_NAME = "FIFA_World_Cup_2026_LR2_2_Data_Simplified.xlsx"
FIG_DIR = "figures_LR2_2"
os.makedirs(FIG_DIR, exist_ok=True)


def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    target_range = np.max(actual) - np.min(actual)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": rmse / target_range,
        "R2": r2_score(actual, predicted),
    }


# 1. LOAD AND CHECK DATA
data_raw = pd.read_excel(FILE_NAME, sheet_name="LR2_2_Data")

explanatory = [
    "Team_Elo_Pre", "Opp_Elo_Pre", "Team_GF_L10", "Opp_GA_L10",
    "Team_WinRate_L10", "Rest_Days", "Home_Nation", "Knockout"
]
target = "GoalsScored"

print("Number of rows:", data_raw.shape[0])
print("Unique matches:", data_raw["MatchID"].nunique())
print("Number of explanatory variables:", len(explanatory))
print("Missing values:")
print(data_raw[explanatory + [target]].isna().sum())
print("Duplicate team-match rows:", data_raw.duplicated(subset=["MatchID", "Team"]).sum())

data = data_raw.dropna(subset=explanatory + [target]).copy()
print("Rows after cleaning:", len(data))


# 2. DESCRIPTIVE STATISTICS AND EXPLORATION
print("\nDESCRIPTIVE STATISTICS")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(data[explanatory + [target]].describe().round(2))

max_rest_row = data.loc[data["Rest_Days"].idxmax(), ["Team", "Opponent", "Rest_Days", "MatchDate"]]
print("\nMaximum rest-days observation:")
print(max_rest_row.to_string())

plt.figure(figsize=(6, 4))
plt.hist(data[target], bins=np.arange(data[target].min() - 0.5, data[target].max() + 1.5, 1), edgecolor="white")
plt.title("Distribution of Goals Scored")
plt.xlabel("Goals Scored")
plt.ylabel("Number of Team-Matches")
save_figure("LR2_2_goals_histogram.png")

plt.figure(figsize=(6, 4))
plt.scatter(data["Team_Elo_Pre"], data[target], alpha=0.7)
plt.title("Team Elo Rating and Goals Scored")
plt.xlabel("Team Elo Rating Before Match")
plt.ylabel("Goals Scored")
save_figure("LR2_2_team_elo_vs_goals.png")

plt.figure(figsize=(6, 4))
plt.scatter(data["Team_GF_L10"], data[target], alpha=0.7)
plt.title("Recent Scoring Form and Goals Scored")
plt.xlabel("Average Goals in Previous 10 Matches")
plt.ylabel("Goals Scored")
save_figure("LR2_2_recent_scoring_vs_goals.png")

correlation = data[explanatory + [target]].corr()
print("\nCorrelation with Goals Scored:")
print(correlation[target].drop(target).round(2))

plt.figure(figsize=(9, 7))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Variables (LR2.2)")
save_figure("LR2_2_correlation_heatmap.png")


# 3. MATCH-ID TRAIN AND TEST SPLIT
# Both team rows from one match stay together to prevent match leakage.
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
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Matches appearing in both sets:", len(set(train_data["MatchID"]) & set(test_data["MatchID"])))


# 4. BASELINE AND MULTIPLE LINEAR REGRESSION
baseline_value = y_train.mean()
baseline_predictions = np.ones(len(y_test)) * baseline_value
baseline_scores = evaluate(y_test, baseline_predictions)

model = LinearRegression()
model.fit(X_train, y_train)
predicted_goals = model.predict(X_test)
model_scores = evaluate(y_test, predicted_goals)

print("\nBASELINE MODEL")
print("Baseline prediction:", round(baseline_value, 3))
for name, value in baseline_scores.items():
    print(name + ":", round(value, 3))

print("\nLINEAR REGRESSION PERFORMANCE")
for name, value in model_scores.items():
    print(name + ":", round(value, 3))

coefficients = pd.DataFrame({"Variable": explanatory, "Coefficient": model.coef_})
print("\nMODEL COEFFICIENTS (training data)")
print(coefficients.round(4))
print("Intercept:", round(model.intercept_, 4))

results_table = pd.DataFrame({"Baseline": baseline_scores, "Linear Regression": model_scores}).T
print("\nBASELINE vs REGRESSION (test set)")
print(results_table.round(3))


# 5. REPEATED 5-FOLD CROSS-VALIDATION
X = data[explanatory]
y = data[target]
scoring = {"RMSE": "neg_root_mean_squared_error", "R2": "r2"}
repeated_cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=1)


def repeated_cv_scores(estimator):
    scores = cross_validate(estimator, X, y, cv=repeated_cv, scoring=scoring)
    return {
        "Mean RMSE": -scores["test_RMSE"].mean(),
        "Mean R2": scores["test_R2"].mean(),
        "R2 SD": scores["test_R2"].std(),
    }


repeated_cv_table = pd.DataFrame({
    "Baseline": repeated_cv_scores(DummyRegressor(strategy="mean")),
    "Linear Regression": repeated_cv_scores(LinearRegression()),
}).T
print("\nREPEATED 5-FOLD CROSS-VALIDATION (5 folds x 20 repeats)")
print(repeated_cv_table.round(3))


# 6. GROUPED 5-FOLD CROSS-VALIDATION BY MATCH ID
group_cv = GroupKFold(n_splits=5)


def group_cv_scores(estimator):
    scores = cross_validate(
        estimator, X, y, groups=data["MatchID"], cv=group_cv, scoring=scoring
    )
    return {
        "Mean RMSE": -scores["test_RMSE"].mean(),
        "Mean R2": scores["test_R2"].mean(),
        "R2 SD": scores["test_R2"].std(),
    }


group_cv_table = pd.DataFrame({
    "Baseline": group_cv_scores(DummyRegressor(strategy="mean")),
    "Linear Regression": group_cv_scores(LinearRegression()),
}).T
print("\nGROUPED 5-FOLD CROSS-VALIDATION BY MATCH ID")
print(group_cv_table.round(3))


# 7. OLS ON ALL 208 ROWS
X_all = np.column_stack([np.ones(len(X)), X.values])
beta, _, _, _ = np.linalg.lstsq(X_all, y.values, rcond=None)
residuals = y.values - X_all @ beta
n, k = X_all.shape
sigma2 = residuals @ residuals / (n - k)
std_errors = np.sqrt(np.diag(sigma2 * np.linalg.inv(X_all.T @ X_all)))
t_values = beta / std_errors
p_values = 2 * stats.t.sf(np.abs(t_values), n - k)
r2_all = 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))
adj_r2 = 1 - (1 - r2_all) * (n - 1) / (n - k)
f_stat = (r2_all / (k - 1)) / ((1 - r2_all) / (n - k))
f_p = stats.f.sf(f_stat, k - 1, n - k)

ols_table = pd.DataFrame(
    {"Coefficient": beta, "Std Error": std_errors, "t": t_values, "p-value": p_values},
    index=["Intercept"] + explanatory,
)
print("\nOLS ON ALL TEAM-MATCH ROWS")
print(ols_table.round(4))
print("R2:", round(r2_all, 3), "Adjusted R2:", round(adj_r2, 3),
      "F:", round(f_stat, 2), "F-test p-value:", round(f_p, 6))


# 8. VARIANCE INFLATION FACTORS
vif = {}
for variable in explanatory:
    others = [item for item in explanatory if item != variable]
    variable_r2 = LinearRegression().fit(X[others], X[variable]).score(X[others], X[variable])
    vif[variable] = 1 / (1 - variable_r2)

print("\nVARIANCE INFLATION FACTORS")
print(pd.Series(vif).round(2))


# 9. ACTUAL VS PREDICTED AND RESIDUAL FIGURES
plt.figure(figsize=(6, 5))
plt.scatter(y_test, predicted_goals, alpha=0.8)
limits = [min(y_test.min(), predicted_goals.min()) - 0.25,
          max(y_test.max(), predicted_goals.max()) + 0.25]
plt.plot(limits, limits, color="red", linestyle="--", label="Perfect prediction (y = x)")
plt.title("Actual vs Predicted Goals (test set)")
plt.xlabel("Actual Goals")
plt.ylabel("Predicted Goals")
plt.legend()
save_figure("LR2_2_actual_vs_predicted.png")

plt.figure(figsize=(6, 4))
plt.scatter(predicted_goals, y_test - predicted_goals, alpha=0.8)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted (test set)")
plt.xlabel("Predicted Goals")
plt.ylabel("Residual (Actual - Predicted)")
save_figure("LR2_2_residuals.png")


# 10. SAVE REPRODUCIBLE OUTPUTS
prediction_table = pd.DataFrame({"Actual": y_test.values, "Predicted": predicted_goals})
prediction_table.to_csv("LR2_2_actual_vs_predicted.csv", index=False)

with pd.ExcelWriter("LR2_2_results.xlsx") as writer:
    results_table.round(4).to_excel(writer, sheet_name="Test_Set")
    repeated_cv_table.round(4).to_excel(writer, sheet_name="Repeated_CV")
    group_cv_table.round(4).to_excel(writer, sheet_name="Grouped_CV")
    ols_table.round(6).to_excel(writer, sheet_name="OLS_Coefficients")
    pd.Series(vif, name="VIF").round(4).to_excel(writer, sheet_name="VIF")

summary = {
    "rows": int(len(data)), "matches": int(data["MatchID"].nunique()),
    "predictors": int(len(explanatory)), "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)), "train_matches": int(train_data["MatchID"].nunique()),
    "test_matches": int(test_data["MatchID"].nunique()),
    "maximum_rest_days": float(max_rest_row["Rest_Days"]),
    "maximum_rest_team": str(max_rest_row["Team"]),
    "strongest_target_correlation_variable": correlation[target].drop(target).abs().idxmax(),
    "strongest_target_correlation": float(correlation.loc[correlation[target].drop(target).abs().idxmax(), target]),
    "baseline": {key: float(value) for key, value in baseline_scores.items()},
    "regression": {key: float(value) for key, value in model_scores.items()},
    "repeated_cv_baseline": {key: float(value) for key, value in repeated_cv_table.loc["Baseline"].items()},
    "repeated_cv_regression": {key: float(value) for key, value in repeated_cv_table.loc["Linear Regression"].items()},
    "group_cv_baseline": {key: float(value) for key, value in group_cv_table.loc["Baseline"].items()},
    "group_cv_regression": {key: float(value) for key, value in group_cv_table.loc["Linear Regression"].items()},
    "ols_r2": float(r2_all), "ols_adjusted_r2": float(adj_r2),
    "ols_f_p_value": float(f_p),
    "ols_coefficients": {
        name: {"coefficient": float(ols_table.loc[name, "Coefficient"]),
               "p_value": float(ols_table.loc[name, "p-value"])}
        for name in ols_table.index
    },
    "vif": {name: float(value) for name, value in vif.items()},
}

with open("LR2_2_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)


# 11. FINAL CHECK
print("\nFINAL CHECK")
print("208 team-match rows:", len(data) == 208)
print("104 unique matches:", data["MatchID"].nunique() == 104)
print("8 explanatory variables:", len(explanatory) == 8)
print("Missing model values:", data[explanatory + [target]].isna().sum().sum())
if model_scores["RMSE"] < baseline_scores["RMSE"]:
    print("The regression model beat the baseline on the grouped holdout test set.")
else:
    print("The regression model did not beat the baseline on the grouped holdout test set.")

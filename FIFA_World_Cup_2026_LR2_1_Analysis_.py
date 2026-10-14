# FIFA WORLD CUP 2026 - LINEAR REGRESSION 2.1
# Predicting the goal difference (Team A goals - Team B goals) of each match
#
# Files needed in the same folder:
#   FIFA_World_Cup_2026_LR2_1_Data.xlsx
#     - Raw_Matches  : 104 match results (score after 90 min + extra time, no shootout goals)
#     - FIFA_Ranking : FIFA Men's World Ranking of 19 November 2025 for all 48 teams
#     - LR2_1_Data   : compiled dataset (104 rows, 8 explanatory variables + target)
# Figures are saved to the folder "figures_LR2_1".

# IMPORT LIBRARIES
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE_NAME = "FIFA_World_Cup_2026_LR2_1_Data.xlsx"
FIG_DIR = "figures_LR2_1"
os.makedirs(FIG_DIR, exist_ok=True)


def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300)
    plt.show()
    plt.close()


# 1. LOAD DATA
raw = pd.read_excel(FILE_NAME, sheet_name="Raw_Matches")
ranking = pd.read_excel(FILE_NAME, sheet_name="FIFA_Ranking")
data = pd.read_excel(FILE_NAME, sheet_name="LR2_1_Data")

print("Raw matches:", raw.shape)
print("FIFA ranking teams:", ranking.shape[0])
print("Compiled dataset:", data.shape)


# 2. DATA WRANGLING IN PYTHON
# Rebuild every derived variable from the raw results and the FIFA ranking,
# using only information available BEFORE each match.
raw = raw.sort_values(["Date", "Match_ID"]).reset_index(drop=True)
raw["Goal_Difference"] = raw["Team_A_Goals"] - raw["Team_B_Goals"]

# FIFA ranking: published 19 Nov 2025, months before the tournament
rank_lookup = dict(zip(ranking["Team"], ranking["FIFA_Rank_Nov2025"]))
missing_teams = (set(raw["Team_A"]) | set(raw["Team_B"])) - set(rank_lookup)
print("Teams without a FIFA rank:", missing_teams if missing_teams else "none")

raw["Rank_A"] = raw["Team_A"].map(rank_lookup)
raw["Rank_B"] = raw["Team_B"].map(rank_lookup)
raw["FIFA_Rank_Diff"] = raw["Rank_A"] - raw["Rank_B"]                       # used for comparison only
raw["FIFA_LogRank_Diff"] = np.log(raw["Rank_A"]) - np.log(raw["Rank_B"])    # final variable


def team_form(previous_matches, match_date):
    """Form of one team from its earlier 2026 World Cup matches (zeros if none yet)."""
    if len(previous_matches) == 0:
        return {"goals": 0, "draw": 0, "clean": 0, "ppm": 0, "gd": 0, "rest": 0}
    gf = np.array([m["gf"] for m in previous_matches])
    ga = np.array([m["ga"] for m in previous_matches])
    points = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    return {"goals": gf.mean(),
            "draw": np.mean(gf == ga),
            "clean": np.mean(ga == 0),
            "ppm": points.mean(),
            "gd": (gf - ga).mean(),
            "rest": (match_date - previous_matches[-1]["date"]).days}


history = {}
form_rows = []
for _, match in raw.iterrows():
    a = team_form(history.get(match["Team_A"], []), match["Date"])
    b = team_form(history.get(match["Team_B"], []), match["Date"])
    opening = match["Team_A"] not in history and match["Team_B"] not in history
    form_rows.append({"Match_ID": match["Match_ID"],
                      "Opening_Match": opening,
                      "Avg_Goals_Scored_Diff": a["goals"] - b["goals"],
                      "Draw_Rate_Diff": a["draw"] - b["draw"],
                      "Clean_Sheet_Rate_Diff": a["clean"] - b["clean"],
                      "Points_Per_Match_Diff": a["ppm"] - b["ppm"],
                      "Avg_Goal_Diff": a["gd"] - b["gd"],
                      "Rest_Days_Diff": a["rest"] - b["rest"]})
    # add this match to each team's history only AFTER its features are built
    history.setdefault(match["Team_A"], []).append(
        {"date": match["Date"], "gf": match["Team_A_Goals"], "ga": match["Team_B_Goals"]})
    history.setdefault(match["Team_B"], []).append(
        {"date": match["Date"], "gf": match["Team_B_Goals"], "ga": match["Team_A_Goals"]})

built = raw.merge(pd.DataFrame(form_rows), on="Match_ID")

# Host status and knockout stage come from the compiled dataset (venue information)
built = built.merge(data[["Match_ID", "Host_Status", "Knockout_Match"]], on="Match_ID")
built = built.sort_values("Match_ID").reset_index(drop=True)

# EXPLANATORY VARIABLES AND TARGET
explanatory = [
    "FIFA_LogRank_Diff",
    "Draw_Rate_Diff",
    "Clean_Sheet_Rate_Diff",
    "Points_Per_Match_Diff",
    "Avg_Goal_Diff",
    "Rest_Days_Diff",
    "Host_Status",
    "Knockout_Match"
]
target = "Goal_Difference"

# Check that the Python-built variables match the compiled Excel dataset
data = data.sort_values("Match_ID").reset_index(drop=True)
max_gap = (built[explanatory + [target]] - data[explanatory + [target]]).abs().max().max()
print("Largest difference between Python-built and Excel dataset:", round(max_gap, 10))


# 3. DATA CLEANING AND CHECKS
model_data = built.dropna(subset=explanatory + [target])
print("\nRows after cleaning:", len(model_data))
print("Missing values in model columns:", model_data[explanatory + [target]].isna().sum().sum())
print("Duplicate matches:", model_data["Match_ID"].duplicated().sum())
print("Number of explanatory variables:", len(explanatory))
print("Opening matches (no earlier World Cup form, form variables = 0):", int(model_data["Opening_Match"].sum()))


# 4. DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(model_data[explanatory + [target]].describe().round(2))

print("\nAverage goal difference:", round(model_data[target].mean(), 2))
print("Goal difference standard deviation:", round(model_data[target].std(), 2))
print("Minimum goal difference:", model_data[target].min())
print("Maximum goal difference:", model_data[target].max())
print("Share of matches where Team A had the better FIFA rank:",
      round((model_data["FIFA_LogRank_Diff"] < 0).mean(), 2))

# HISTOGRAM OF TARGET
plt.figure(figsize=(6, 4))
plt.hist(model_data[target], bins=np.arange(model_data[target].min() - 0.5, model_data[target].max() + 1.5, 1),
         edgecolor="white")
plt.title("Distribution of Goal Difference")
plt.xlabel("Goal Difference (Team A - Team B)")
plt.ylabel("Number of Matches")
save_figure("LR2_1_goal_difference_histogram.png")

# FIFA RANK vs GOAL DIFFERENCE
plt.figure(figsize=(6, 4))
plt.scatter(model_data["FIFA_LogRank_Diff"], model_data[target], alpha=0.7)
slope, intercept = np.polyfit(model_data["FIFA_LogRank_Diff"], model_data[target], 1)
x_line = np.linspace(model_data["FIFA_LogRank_Diff"].min(), model_data["FIFA_LogRank_Diff"].max(), 50)
plt.plot(x_line, intercept + slope * x_line, color="red")
plt.title("FIFA Rank Difference and Goal Difference")
plt.xlabel("ln(Rank A) - ln(Rank B)   (negative = Team A ranked higher)")
plt.ylabel("Goal Difference")
save_figure("LR2_1_fifa_rank_vs_goal_difference.png")


# 5. CORRELATION
correlation = model_data[explanatory + [target]].corr()
print("\nCorrelation with Goal Difference:")
print(correlation[target].drop(target).round(2))

plt.figure(figsize=(9, 7))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Variables (LR2.1)")
save_figure("LR2_1_correlation_heatmap.png")


# 6. TRAIN AND TEST DATA
X = model_data[explanatory]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    return {"MAE": mean_absolute_error(actual, predicted),
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": rmse / (np.max(actual) - np.min(actual)),
            "R2": r2_score(actual, predicted)}


# 7. BASELINE MODEL (always predicts the training mean)
baseline_value = y_train.mean()
baseline_predictions = np.ones(len(y_test)) * baseline_value
baseline_scores = evaluate(y_test, baseline_predictions)
print("\nBASELINE MODEL")
print("Baseline prediction:", round(baseline_value, 3))
for name, value in baseline_scores.items():
    print(name + ":", round(value, 3))


# 8. MULTIPLE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)
predicted_goal_difference = model.predict(X_test)
model_scores = evaluate(y_test, predicted_goal_difference)

print("\nLINEAR REGRESSION PERFORMANCE")
for name, value in model_scores.items():
    print(name + ":", round(value, 3))

coefficients = pd.DataFrame({"Variable": explanatory, "Coefficient": model.coef_})
print("\nMODEL COEFFICIENTS (training data)")
print(coefficients.round(3))
print("Intercept:", round(model.intercept_, 3))

results_table = pd.DataFrame({"Baseline": baseline_scores, "Linear Regression": model_scores}).T
print("\nBASELINE vs REGRESSION (test set)")
print(results_table.round(3))


# 9. ROBUSTNESS CHECK: REPEATED 5-FOLD CROSS-VALIDATION
# A single 80/20 split tests only 21 matches, so results depend heavily on which
# matches land in the test set. Repeated cross-validation gives a fairer estimate.
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=1)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def cross_validate_model(estimator, features):
    scores = cross_validate(estimator, model_data[features], y, cv=cv, scoring=scoring)
    return {"CV MAE": -scores["test_MAE"].mean(),
            "CV RMSE": -scores["test_RMSE"].mean(),
            "CV R2": scores["test_R2"].mean(),
            "CV R2 SD": scores["test_R2"].std()}


cv_table = pd.DataFrame({
    "Baseline": cross_validate_model(DummyRegressor(strategy="mean"), explanatory),
    "Linear Regression": cross_validate_model(LinearRegression(), explanatory)}).T
print("\nREPEATED 5-FOLD CROSS-VALIDATION (5 folds x 20 repeats)")
print(cv_table.round(3))


# 10. MODEL DEVELOPMENT: WHY THE FIFA RANK VARIABLE WAS ADDED
# Initial model: in-tournament form only (with Avg_Goals_Scored_Diff).
# Improved models: Avg_Goals_Scored_Diff replaced by a pre-tournament FIFA rank difference.
initial_variables = ["Avg_Goals_Scored_Diff"] + explanatory[1:]
rank_variables = ["FIFA_Rank_Diff"] + explanatory[1:]
development = pd.DataFrame({
    "1. Initial (in-tournament form only)": cross_validate_model(LinearRegression(), initial_variables),
    "2. + FIFA rank difference": cross_validate_model(LinearRegression(), rank_variables),
    "3. + log FIFA rank difference (final)": cross_validate_model(LinearRegression(), explanatory)}).T
print("\nMODEL DEVELOPMENT (repeated 5-fold CV)")
print(development.round(3))


# 11. STATISTICAL SIGNIFICANCE (OLS ON ALL 104 MATCHES)
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

ols_table = pd.DataFrame({"Coefficient": beta, "Std Error": std_errors, "t": t_values, "p-value": p_values},
                         index=["Intercept"] + explanatory)
print("\nOLS ON ALL MATCHES")
print(ols_table.round(3))
print("R2:", round(r2_all, 3), " Adjusted R2:", round(adj_r2, 3),
      " F:", round(f_stat, 2), " F-test p-value:", round(f_p, 5))

# MULTICOLLINEARITY (VARIANCE INFLATION FACTOR)
vif = {}
for variable in explanatory:
    others = [v for v in explanatory if v != variable]
    r2_v = LinearRegression().fit(X[others], X[variable]).score(X[others], X[variable])
    vif[variable] = 1 / (1 - r2_v)
print("\nVARIANCE INFLATION FACTORS")
print(pd.Series(vif).round(2))


# 12. ACTUAL VS PREDICTED
plt.figure(figsize=(6, 5))
plt.scatter(y_test, predicted_goal_difference, alpha=0.8)
limits = [min(y_test.min(), predicted_goal_difference.min()) - 0.5,
          max(y_test.max(), predicted_goal_difference.max()) + 0.5]
plt.plot(limits, limits, color="red", linestyle="--", label="Perfect prediction (y = x)")
plt.title("Actual vs Predicted Goal Difference (test set)")
plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.legend()
save_figure("LR2_1_actual_vs_predicted.png")

# RESIDUALS
plt.figure(figsize=(6, 4))
plt.scatter(predicted_goal_difference, y_test - predicted_goal_difference, alpha=0.8)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted (test set)")
plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual (Actual - Predicted)")
save_figure("LR2_1_residuals.png")


# 13. SAVE RESULTS FOR THE REPORT
prediction_table = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predicted_goal_difference
})
prediction_table.to_csv("LR2_1_actual_vs_predicted.csv", index=False)

with pd.ExcelWriter("LR2_1_results.xlsx") as writer:
    results_table.round(3).to_excel(writer, sheet_name="Test_Set")
    cv_table.round(3).to_excel(writer, sheet_name="Cross_Validation")
    development.round(3).to_excel(writer, sheet_name="Model_Development")
    ols_table.round(4).to_excel(writer, sheet_name="OLS_Coefficients")
    pd.Series(vif, name="VIF").round(2).to_excel(writer, sheet_name="VIF")

summary = {
    "rows": int(len(model_data)),
    "predictors": int(len(explanatory)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "opening_matches": int(model_data["Opening_Match"].sum()),
    "team_a_better_rank_share": float((model_data["FIFA_LogRank_Diff"] < 0).mean()),
    "strongest_target_correlation_variable": correlation[target].drop(target).abs().idxmax(),
    "strongest_target_correlation": float(correlation.loc[correlation[target].drop(target).abs().idxmax(), target]),
    "baseline": {key: float(value) for key, value in baseline_scores.items()},
    "regression": {key: float(value) for key, value in model_scores.items()},
    "cv_baseline": {key: float(value) for key, value in cv_table.loc["Baseline"].items()},
    "cv_regression": {key: float(value) for key, value in cv_table.loc["Linear Regression"].items()},
    "ols_r2": float(r2_all),
    "ols_adjusted_r2": float(adj_r2),
    "ols_f_p_value": float(f_p),
    "ols_coefficients": {
        name: {
            "coefficient": float(ols_table.loc[name, "Coefficient"]),
            "p_value": float(ols_table.loc[name, "p-value"])
        }
        for name in ols_table.index
    },
    "vif": {name: float(value) for name, value in vif.items()}
}

with open("LR2_1_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)


# 14. FINAL CHECK
print("\nFINAL CHECK")
print("104 matches:", len(model_data) == 104)
print("8 explanatory variables:", len(explanatory) == 8)
print("Missing model values:", model_data[explanatory + [target]].isna().sum().sum())
if model_scores["RMSE"] < baseline_scores["RMSE"] and cv_table.loc["Linear Regression", "CV RMSE"] < cv_table.loc["Baseline", "CV RMSE"]:
    print("The regression model beat the baseline on the test set and in cross-validation.")
else:
    print("The regression model did not consistently beat the baseline.")

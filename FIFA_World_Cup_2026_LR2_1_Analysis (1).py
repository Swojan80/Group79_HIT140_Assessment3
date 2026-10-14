# FIFA WORLD CUP 2026 - LINEAR REGRESSION 2.1
# Predicting goal difference between two teams

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
data = pd.read_excel("FIFA_World_Cup_2026_LR2_1_Data.xlsx", sheet_name="LR2_1_Data")

print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))
print("\nMissing values:")
print(data.isna().sum())

# EXPLANATORY VARIABLES
explanatory = [
    "Avg_Goals_Scored_Diff",
    "Draw_Rate_Diff",
    "Clean_Sheet_Rate_Diff",
    "Points_Per_Match_Diff",
    "Avg_Goal_Diff",
    "Rest_Days_Diff",
    "Host_Status",
    "Knockout_Match"
]

target = "Goal_Difference"

print("\nNumber of explanatory variables:", len(explanatory))
print("Number of matches:", data["Match_ID"].nunique())

# DATA CLEANING
data_clean = data.dropna(subset=explanatory + [target])

print("Rows after cleaning:", len(data_clean))
print("Duplicate matches:", data_clean["Match_ID"].duplicated().sum())

# DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")
print(data_clean[explanatory + [target]].describe().round(2))

print("\nAverage goal difference:", round(data_clean[target].mean(), 2))
print("Goal difference standard deviation:", round(data_clean[target].std(), 2))
print("Minimum goal difference:", data_clean[target].min())
print("Maximum goal difference:", data_clean[target].max())

# HISTOGRAM
plt.hist(data_clean[target], bins=11)
plt.title("Distribution of Goal Difference")
plt.xlabel("Goal Difference")
plt.ylabel("Number of Matches")
plt.show()

# CORRELATION
correlation = data_clean[explanatory + [target]].corr()

print("\nCorrelation with Goal Difference:")
print(correlation[target].drop(target).round(2))

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Between Variables")
plt.show()

# PREPARE MODEL DATA
X = data_clean[explanatory]
y = data_clean[target]

# TRAIN AND TEST DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# BASELINE MODEL
baseline_value = y_train.mean()
baseline_predictions = np.ones(len(y_test)) * baseline_value

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_rmse = sqrt(baseline_mse)

print("\nBASELINE MODEL")
print("Baseline prediction:", round(baseline_value, 3))
print("MAE:", round(baseline_mae, 3))
print("MSE:", round(baseline_mse, 3))
print("RMSE:", round(baseline_rmse, 3))

# BUILD MULTIPLE LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# MAKE PREDICTIONS
predicted_goal_difference = model.predict(X_test)

# MODEL PERFORMANCE
mae = mean_absolute_error(y_test, predicted_goal_difference)
mse = mean_squared_error(y_test, predicted_goal_difference)
rmse = sqrt(mse)
r2 = r2_score(y_test, predicted_goal_difference)

print("\nLINEAR REGRESSION PERFORMANCE")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))

# COEFFICIENTS
coefficients = pd.DataFrame({"Variable": explanatory,"Coefficient": model.coef_})

print("\nMODEL COEFFICIENTS")
print(coefficients.round(3))
print("Intercept:", round(model.intercept_, 3))

# COMPARE WITH BASELINE
print("\nMODEL COMPARISON")
print("Baseline RMSE:", round(baseline_rmse, 3))
print("Regression RMSE:", round(rmse, 3))

if rmse < baseline_rmse:
    print("The regression model performed better than the baseline.")
else:
    print("The baseline performed better than the regression model.")

# ACTUAL VS PREDICTED
plt.scatter(y_test, predicted_goal_difference)
plt.title("Actual vs Predicted Goal Difference")
plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.show()

# FINAL CHECK
print("\nFINAL CHECK")
print("104 matches:", len(data_clean) == 104)
print("8 explanatory variables:", len(explanatory) == 8)
print("Missing model values:", data_clean[explanatory + [target]].isna().sum().sum())

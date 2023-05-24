import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from data import features_df

# Separate features and target variable
x = features_df.iloc[:, :6]
y = features_df.iloc[:, 6]

# Data Splitting in test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# MLR Model Creation
mlr = LinearRegression()
mlr.fit(x_train, y_train)

# Coefficients
coef_names = features_df.columns[:6]
coef_values = mlr.coef_
print(list(zip(coef_names, coef_values)))

# Predictions
y_pred_mlr = mlr.predict(x_test)
print("Prediction for test set: {}".format(y_pred_mlr))

# Comparing Actual and Predictions
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr}).astype(int)
print(mlr_diff.head())

# Model Evaluation
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(meanSqErr)
r_squared = mlr.score(x, y) * 100

print('R squared: {:.2f}'.format(r_squared))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

# Plotting actual/predicted values
plt.figure(figsize=(10, 6))
plt.plot(mlr_diff.index, mlr_diff['Actual value'], marker='o', linestyle='-', label='Actual')
plt.plot(mlr_diff.index, mlr_diff['Predicted value'], marker='o', linestyle='-', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
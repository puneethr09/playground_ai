"""
Linear Regression Example
This script demonstrates a simple linear regression model using synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Parameters
NUM_SAMPLES = 100
NOISE_LEVEL = 2
TEST_SIZE = 0.2

# Generate some sample data
np.random.seed(0)
X = np.random.rand(NUM_SAMPLES, 1) * 10  # 100 rows, 1 column
y = 2.5 * X + np.random.randn(NUM_SAMPLES, 1) * NOISE_LEVEL  # Linear relation with some noise

# Convert to DataFrame
data = pd.DataFrame(data=np.hstack((X, y)), columns=['Feature', 'Target'])

# Display the first few rows of the DataFrame
print(data.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Feature']], data['Target'], test_size=TEST_SIZE, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Example')
plt.grid()
plt.legend()
plt.savefig('linear_regression_plot.png')  # Save the plot
plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 100 rows, 1 column
y = 2.5 * X + np.random.randn(100, 1) * 2  # Linear relation with some noise

# Convert to DataFrame
data = pd.DataFrame(data=np.hstack((X, y)), columns=['Feature', 'Target'])

# Display the first few rows of the DataFrame
print(data.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Feature']], data['Target'], test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

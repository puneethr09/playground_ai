# Import necessary libraries
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
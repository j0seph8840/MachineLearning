import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in the "carbig" dataset and clean any rows.
df = pd.read_excel('proj1Dataset.xlsx')
df = df.dropna()
print(df)

# Select 'Weight' as predcitor and 'Horsepower' as target variable
predictor = np.array(df['Weight'])
target = np.array(df['Horsepower'])

# Add a column of ones to X for the bias term
X = np.c_[predictor, np.ones_like(predictor)]
print(X)

# Closed-form solution
w = np.linalg.pinv(X) @ target
print(w)

# Visualize the data and regression line
plt.scatter(predictor, target, marker='x', label='Data points', color='red')
plt.plot(predictor, X @ w, label='Closed-form solution', color='blue')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()
plt.show()
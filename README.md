# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the salary dataset
2. Separate input features and target values, then convert categorical data into numerical form
3. Split the dataset into training and testing data, and train the Decision Tree Regressor model
4. Plot and display the Decision Tree Regressor using plot_tree()

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Renita Ireen G
RegisterNumber: 212225220083 
*/import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.read_csv("/content/sample_data/Salary.csv")

X = df[['Level']]

y = df['Salary']

model = DecisionTreeRegressor(random_state=42)

model.fit(X, y)

prediction = model.predict([[6.5]])

print("Predicted Salary:", prediction[0])

plt.scatter(df['Level'], df['Salary'])

plt.plot(df['Level'], model.predict(X))

plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")

plt.show()

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=['Level'],
          filled=True)

plt.title("Decision Tree Regressor Tree")

plt.show()

```

## Output:

<img width="945" height="72" alt="Screenshot 2026-05-25 131329" src="https://github.com/user-attachments/assets/e9db382b-490b-4fd9-ad9f-8098891764b8" />

<img width="843" height="567" alt="Screenshot 2026-05-25 131339" src="https://github.com/user-attachments/assets/adf6d213-ccc3-42e0-a35c-360d70da49ed" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

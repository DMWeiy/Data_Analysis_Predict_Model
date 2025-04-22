import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
data = pd.read_csv(
    "D:\\Pycharm\\IML_Project\\New_Water_Quality_AfterCleaning.csv"
)
df = pd.DataFrame(data)
df = df.sort_values("Timestamp")
print(df)
check = df.isnull().sum()
print(check)

x = df.drop(columns=['Timestamp', 'pH'])

print(x)

y = df["pH"]

print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)

c = lr.intercept_

y_pred = lr.predict(x_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(r2)
print(rmse)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="Actual Test Data", color="g")
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Data", color="b")
plt.title("Actual vs Predicted pH Levels")
plt.xlabel("Sample Index")
plt.ylabel("pH Level")
plt.legend()
plt.show()






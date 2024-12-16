import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

cars=pd.read_csv("cars.csv")
print(cars.head())
print(cars.columns)
plt.figure(figuresize = (16,8))
plt.scatter(
    cars['Horsepower'],
    cars['Price in thousands'],
    c = 'black'
)
plt.xlabel=("horsepower")
plt.ylabel=("price")
plt.show()

X = cars['Horsepower'].values.reshape(-1,1)
Y = cars['Price in thousands'].values.reshape(-1,1)

reg= LinearRegression()
reg.fit(X,Y)
print(reg.coef_[0][0])
print(reg.intercept_[0])

predictions = reg.predict(X)
plt.figure(figuresize=(16,8))
plt.scatter(
    cars['Horsepower'],
    cars['Price in thousands'],
    c = 'black'
)
plt.plot(
    cars['Horsepower'],
    predictions,
    c = 'blue',
    linewidth = 2
)
plt.xlabel("Horsepower")
plt.ylabel("prices")
plt.show()

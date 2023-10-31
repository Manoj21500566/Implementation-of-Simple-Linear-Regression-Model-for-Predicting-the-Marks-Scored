# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
Step 1: Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the simple linear regression model for predicting the marks scored.

Step 2: Set variables for assigning dataset values and implement the .iloc module for slicing the values of the variables X and y.

Step 3: Import the following modules for linear regression; from sklearn.model_selection import train_test_split and also from sklearn.linear_model import LinearRegression.

Step 4: Assign the points for representing the points required for the fitting of the straight line in the graph.

Step 5: Predict the regression of the straight line for marks by using the representation of the graph.

Step 6: Compare the graphs (Training set, Testing set) and hence we obtained the simple linear regression model for predicting the marks scored using the given datas.

Step 7: End the program. 

## Program
~~~

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Manoj M
RegisterNumber:  212221240027
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)



~~~

## Output:

df.head()

![1](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/cc49237f-0268-4469-a538-18b561ead863)

df.tail()

![2](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/055f5082-437c-4cbf-850d-85d6d741caa8)

Array value of X

![3](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/505285b5-fb85-491b-9214-6d421a672c46)

Array value of Y

![4](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/5075e036-adaf-48f0-bbd7-9d4fbe171442)

Values of Y prediction

![5](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/db7a5828-b641-4d39-8f35-c70ca6cc5de6)

Array values of Y test

![6](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/75f67356-2dc6-4c22-b7ca-08494c33b9f1)

Training Set Graph

![7](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/bff84470-9b9f-4f78-b848-c66ac82fa176)

Test Set Graph

![8](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/e5fcf953-1d8b-457a-99fc-b863b6770ca1)

Values of MSE, MAE and RMSE

![9](https://github.com/Manoj21500566/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94588708/0e752907-7d3a-4b5b-9b71-a6db5aad7498)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


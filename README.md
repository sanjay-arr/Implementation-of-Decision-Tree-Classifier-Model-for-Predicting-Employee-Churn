# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: G.SANJAY
RegisterNumber: 212224230243 
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### DATA HEAD:
![image](https://github.com/user-attachments/assets/f78abd2a-1e84-4dd2-a8f7-562c18d8410f)

<br>
<br>
<br>
<br>
<br>
<br>

### DATASET INFO:
![image](https://github.com/user-attachments/assets/abd11de8-fd64-4574-8952-54c9f2bf6086)
### NULL DATASET:
![image](https://github.com/user-attachments/assets/acdecc6f-c072-4362-b9b8-3c2341b469e6)
### VALUES COUNT IN LEFT COLUMN:
![image](https://github.com/user-attachments/assets/a3e26432-14bc-41ac-a1a9-258eae21ddc2)
### DATASET TRANSFORMED HEAD:
![image](https://github.com/user-attachments/assets/04c379f0-911b-4c74-bbc0-6e72cb56fcd4)
### X.HEAD:
![image](https://github.com/user-attachments/assets/a24781fe-f112-41df-a591-e059b550423d)
### ACCURACY:
![image](https://github.com/user-attachments/assets/2814107f-063a-467e-b865-e274cceb1fe5)
### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/d4d1abc5-36c4-4a0f-af18-4f7bbf8e35ad)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

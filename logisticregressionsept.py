

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\DELL 3410\OneDrive\ML Projects\DataSet\insurance.csv")

df.head()

df.info()

df.head()

# no of rows and columns

df.shape

# names of columns
df.columns

df.info()

# print random 10 rows from my dataset

df.sample(10)

# different jobs and number of people doing the particular job

df['Anual_Salary'].value_counts()

# 8:42 -- run a loop and print the categories of all object type columns

for col in df.columns:
  if df[col].dtype == 'object':
    print(df[col].value_counts())
    print()

df.isnull().sum()

df.isnull().sum().sum()

df.dropna(inplace = True)

df.shape

# checking the duplicates

df.duplicated().sum()

# remove the duplicates

df.drop_duplicates(inplace = True)

df.shape

for col in df.columns:
  if df[col].dtype !='object':
    plt.boxplot(df[col])
    plt.xlabel(col)
    plt.show()

# #duration  -- how to remove outliers
# 1. calculate q1 and q3
# 2. calculate iqr
# 3. calculate ul and ll
# 4. filter the data on the basis of ul and ll

# iqr -- inter quartile range

q1 = df['duration'].quantile(0.25)

q1

q3 = df['duration'].quantile(0.75)

q3

iqr = q3 - q1

ul = q3 + 1.5*iqr
ll = q1 - 1.5*iqr

ul

ll

df = df[(df['duration']>=ll) & (df['duration']<=ul)]

for col in df.columns:
  if df[col].dtype !='object':
    plt.boxplot(df[col])
    plt.xlabel(col)
    plt.show()

df.info()

"""# Label Encoding"""

# label encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
  if df[col].dtype =='object':
    df[col] = le.fit_transform(df[col])

df.info()

# Clean data --> model building

df.head()

"""
# Model building

"""

x = df.drop('y', axis = 1)
y = df['y']

x=df.iloc[:,:-1]
y = df.iloc[:,-1]

# you can use this if the target column is in the last
#it is based on slicing [:,:-1]--first part says get all rows, second part says all columns except last
# [:,-1] -- get all rows and just the last column

x

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 7)

x_train

x_train.shape

x_test.shape

y_train.shape

y_test.shape

y_train

"""# Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

import warnings
warnings.filterwarnings('ignore')

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

y_pred

y_pred.shape

from sklearn.metrics import *

confusion_matrix(y_test,y_pred)

test_accuracy = accuracy_score(y_test,y_pred)
test_accuracy

precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

# training accuracy

y_train_pred = lr.predict(x_train)

y_train_pred

train_accuracy = accuracy_score(y_train,y_train_pred)

train_accuracy

train_accuracy, test_accuracy

"""# Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

y_pred_dt = dt_model.predict(x_test)

y_pred_dt

dt_a = accuracy_score(y_test,y_pred_dt)

dt_a

y_train_pred_dt = dt_model.predict(x_train)

y_train_pred_dt

tr_a_dt = accuracy_score(y_train,y_train_pred_dt)

tr_a_dt

print("Training accuracy for dt model", tr_a_dt)
print("Testing accuaracy for dt model", dt_a)

# This is an absolute case of overfitting
# Since the data is simpler while we are using a complex model its learning the patterns way to well that it is even
# learning the noise so causing overfitting
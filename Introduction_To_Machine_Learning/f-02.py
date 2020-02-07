import numpy as np
import pandas as pd


def change_sex(str):
    if str=='male':
        return 0
    else:
        return 1


def change_embarked(str):
    if str=='C':
        return 0
    elif str=='Q':
        return 1
    else :
        return 2

x_o=pd.read_csv('G:\Development\Projects\Python Projects\Machine_Learning\Logistic_Regression\Titanic_Project\\0000000000002429_training_titanic_x_y_train.csv')
abc=(x_o.describe())
abc_fixed = abc.reset_index().replace(
    {'25%': '25_percent', '50%': '50_percent', '75%': '75_percent'}).set_index('index')
x1=x_o.copy()
# x1=x[:,0:11]
# y1=x[:,11]
del x1['Name']
del x1['Ticket']
del x1['Cabin']

x1['Gender']=x1.Sex.apply(change_sex)
del x1['Sex']

x1.Age.fillna(x1.Age.mean(),inplace=True)
del x1['Fare']




x1['New_Embarked']=x1.Embarked.apply(change_embarked)
del x1['Embarked']

print(x1)


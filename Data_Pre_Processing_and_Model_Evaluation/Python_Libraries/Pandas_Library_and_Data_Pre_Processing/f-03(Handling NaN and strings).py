# Handling NaN
# For all the null entries we either remove them or fill them with some data entries.
import pandas as pd
import numpy as np
iris=pd.read_csv("G:\Development\Projects\Python Projects\Machine_Learning\Pandas and Matplot Libraries\iris_dataframe.csv",header=None)
df=iris.copy()
df.columns=['sl','sw','pl','pw','ft']
df.head()
# As the current data frame has no NaN entries, therefore we make some NaN entries
df.iloc[2:5,1:4]=np.nan                  # This is a function in numpy(nan)
# Now entries are made NaN.
# Method 1:- Dropping The NaN entries.............................
print(df.dropna())                       # Here NaN entries are dropped but change don't persist.
print(df.dropna(inplace=True))           # Here NaN entries are dropped with changes persisting.
df.reset_index(drop=True,inplace=True)   # Here index is reset.
print(df.head())

# Resetting the data frame.................................
df=iris.copy()
df.columns=['sl','sw','pl','pw','ft']
df.iloc[2:5,1:4]=np.nan                  # Creating NaN entries

# Method 2:- Filling the NaN entries with value(mean of entire column)......................

df.sw.fillna(df.sw.mean(),inplace=True)
# sw-Name of column
# fillna- The function used to fill the NaN entries
# mean- The function used to calculate mean
a=df[df.ft=='Iris-Setosa']
df.pl.fillna(a.pl.mean(),inplace=True)  # For the pl column and only for the flower type iris setosa
print(df.dropna())

# Handling Strings.......................
# Here there are three types of flowers in our current data frame: Iris-setosa ,Iris-virginica and Iris-versicolor
# As numbers are easy to work with we have to assign each string a number.
# Let Iris-setosa=1 ,Iris-virginica=3 and Iris-versicolor=2
# To do this we can use the apply function.It takes a function (definition) as a parameter
def func(str):
    if str=='Iris-setosa':
        return 1
    elif str=='Iris-virginica':
        return 3
    elif str=='Iris-versicolor':
        return 2
    else:
        return 0

df['new_ft']=df.ft.apply(func)
# The func function passed takes each data item and creates a new entry with the corresponding flower number.
print(df.head(150))


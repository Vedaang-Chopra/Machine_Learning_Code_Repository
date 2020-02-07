# Dropping Columns
import pandas as pd
iris=pd.read_csv("G:\Development\Projects\Python Projects\Machine_Learning\Pandas and Matplot Libraries\iris_dataframe.csv",header=None)
df=iris.copy()
df.columns=['sepal_length','sepal_width','petal_length','petal_width','f_type']
df.head()
#print(df.f_type)

# Note:-
# 1. The drop function is used to drop a row from the table.
# 2. It takes the label of data row as parameter.
# 3. The label of any data row is specific position/index number assigned to the row.
# 4. The index number associated with data item is the label.
# 5. The position of the data item changes but the label remains fixed after using the drop function.
# 6. Initially the label and position are same.
# 7. Labels can be changed externally using some other method.

a=df.drop(0)
# The drop function that is df.drop doesn't change the object df(data frame) so object 'a' has to be
# made for storing the new changes.
print(a.head(4))
print(df.head(4))         # df remains intact/previous as before
df.drop(0,inplace=True)
# By using the inplace parameter changes made in df via drop function would persist.
print(df.head())
df.drop(3,inplace=True)   # Changes made in df
print(df.head())

print(df.index)           # Label-Position array
# Note:-The index parameter gives the corresponding label, that is it gives an
# array with all the labels present in the data frame. The position of the label in
# the array is the position of the row in data frame.
df.index[0]                               # Gives the label occurring at first position in the data frame
print(df.index[[0,1,3,9,4,5,7,6]])          # This returns a tuple holding the corresponding labels at the positions
df.drop(df.index[0],inplace=True)         # Deleting single item by position
df.drop(df.index[[0,1]],inplace=True)     # Deleting multiple items by position

# Running conditions on data frame....................
df.sepal_length>5
# This gives a table of label and sepal_length with boolean data where the value of sepal_length>5 and where it is not.
df[df.sepal_length>5]
# This gives a table of label and sepal_length and only displays row where the value of sepal_length>5

c1=df[df.sepal_length>5].describe()       # Conditions can be run on data frame
print(c1)

# Iloc vs Loc and Adding New Row

# Loc-  It is a function which returns data row and takes argument as Label
# ILoc- It is a function which returns data row and takes argument as Position
print(df.loc[10])           # Display data of row having label-10
print(df.iloc[10])          # Display data of row having position-10

# Adding new rows:
# Note:- As no data row with label-0 exists(previously deleted), therefore with loc function we can create a new row.
df.loc[0]=[1,2,3,4,'Vedaang']       # By using loc new row is created as previously no label-0 exists.
# Note:- The new row created is pushed into the last of the table
print(df.tail())
print(df.drop(0,inplace=True))
print(df.drop(7,inplace=True))

# Resetting Index
print(df.reset_index())
# This function resets the previous labels/index by adding a new column of indexes to the previous data frame.
# As the new column is not required so the drop parameter is used.
print(df.reset_index(inplace=True,drop=True))  # The drop=True doesn't create new column and inplace changes the existing data frame.
print(df.index)

# Deleting Columns
print(df.drop('sepal_width',axis=1))           # axis=1 sets to look/move column wise
del df['sepal_length']                         # The del keyword can also be used to delete stuff
print(df.head)

# Resetting the entire database again
df=iris.copy()
df.columns=['sepal_length','sepal_width','petal_length','petal_width','f_type']
print(df.head())

# Adding Columns
# Using this method we can add a column
df['petal_width_and_length_difference']=df['petal_length']-df['petal_width']
print(df.head(6))

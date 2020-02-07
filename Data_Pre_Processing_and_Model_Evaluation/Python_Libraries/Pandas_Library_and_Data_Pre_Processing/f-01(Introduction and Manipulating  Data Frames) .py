# Note:-
# 1. Pandas is a module which is used to manipulate data. The data presented to it is in the form of
#    data sets/tabular form/similar to excel file.
# 2. Panda is used to load, clean and manage data sets.
# 3.  Why is Panda Package used in python ?
#     Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working
#     with “relational” or “labeled” data both easy and intuitive.It aims to be the fundamental high-level building block
#     for doing practical, real world data analysis in Python.
import pandas as pd                 # Importing Pandas Module

# iris=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
iris=pd.read_csv("G:\Development\Projects\Python Projects\Machine_Learning\Pandas and Matplot Libraries\iris_dataframe.csv",header=None)
# Note:-
# 1. The read_csv function is used to read the csv file.
# 2. There is also a read_json, read_html, read_excel function etc.
# 3. A csv file is data storing file where data is stored in tabular form where
#    columns are separated by comma and rows are separated by new line.
# 4. The read_csv function takes path of csv file as argument and returns an object which holds the data frame.
# 5. As when data set is copied the first data entry is treated as column header, therefore
#    we have to change it to proper column headers.
# 6. By using just the above method the first row is set as header.
#    To avoid such thing use header=none parameter in read_csv function.
# print(iris)
print(type(iris))
# Some Important points while Working on a data frame:-
# Note:- Always make a copy of data frame whenever we work on a project.
d=iris                  # Changes made in d are reflected in iris
# Note:-Using the above method the original data frame is just referenced with another name 'd'.
#       A new copy has not been made.
df=iris.copy()          # Changes made in df are not reflected in iris
# Note:- By using the above method a new copy has been made of the original data frame.

# Functions to be used for knowing the data frame attributes:-

print(df.shape)
# The shape function tells the value: no. of rows * no. of columns
df.columns=['sepal_length','sepal_width','petal_length','petal_width','flower_type']
# The columns function is used to change the labels/headers of columns. Initially the first row was treated as labels/fields.
print(df.head(3))       # Prints the starting entries and takes a no. as argument for no. of entries to display ; Default Value-5
df.tail()               # Same as head function ; default value-5
print(df.dtypes)        # Tells the data type for each column
print(df.describe())
# The describe function describes the data set for us. This works only on integer/number column.
# By describing it means that it works some mathematical function on columns and displays result.
# eg:- It runs count,mean etc. functions and displays result.

# Printing Data Frame Column / Accessing particular column of Data Frame
print(df.petal_length)      #Ways to access column
print(df['petal_length'])
print(df.sepal_length)
# This prints the index and column information.

# Finding Null Entries:-
df.isnull()             # Gives a table showing whether an entry is null or not
df.isnull().sum()       # Gives count of nulls in table columns(works for particular column)

# Slicing Data Frame/ Accessing Data from some rows and columns in between Data Frame
df.iloc[1:4,2:5]        # Used to slice particular data from table, first slice for rows and second for columns




# iris.append([])
# for i in range(len(iris),0):

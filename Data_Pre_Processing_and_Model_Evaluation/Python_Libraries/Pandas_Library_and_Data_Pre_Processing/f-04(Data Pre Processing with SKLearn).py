# PreProcessing Data with Sklearn

# Handling NaN enteries......................................
import pandas as pd
import numpy as np
iris=pd.read_csv("iris_dataframe.csv",header=None)
df=iris.copy()
df.columns=['sl','sw','pl','pw','ft']
# As the current data frame has no NaN entries, therefore we make some NaN entries
df.iloc[2:5,1:4]=np.nan                  # This is a function in numpy(nan)
# print(df.head())

# Library for Pre-Processing
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
# 1. The Missing value parameter helps to select some specific value in the column
# 2. The Strategy parameter helps us decide how dow e handle those values, whether to delete it, fill it with mean etc. By default mean is selected.
# 3. The Axis Parameter helps to decide the selection via along row or column. Axis=0 means along column and axis=1 means along row

# Now the Imputer object needs to be fitted with the dataset....................
transformed_data=imputer.fit(df.iloc[:,1:4]).transform(df.iloc[:,1:4])
# We will fit the imputer object will only fit to the NaN values and not all values and transform to form the required data.
# print(transformed_data)
# Then we can substiute it in place of actual data


# Handling categorical Data......................................
# Basically Handling String requires us to convert all our unique strings to numbers or labels which can be done with the help of label encoder.
titanic=pd.read_csv("titanic_train.csv",header=None)
print(titanic.head())

# We use the label encoder to create encodings
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()

# This encodings result in a array of unique numbers which have unique labels inplace of strings.
encoded_columm=label_encoder.fit_transform((titanic.iloc[:,4]))
print(encoded_columm)
# titanic.replace(titanic.iloc[:2],encoded_columm)
titanic['Sex']=encoded_columm
print(titanic.iloc[0:2,:])
# As we are using numbers, the mathematical equations of machine learning algortihms could treat these numbered labels as actual numbers with priority levels.
# Thus to avoid such scenario we use one hot encoding method. Here each label is converted into a vector which represents wheeher the labels value is true or not.

# Using One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[12])
# Note: Always use label encoder before one hot encoder
# Categorical features Parameter is entered so as to specify which column needs to be one hot encoded.
x=onehotencoder.fit_transform(titanic[:]).toarray()
print(x)



# Note: Label Binarizer Uses Both One Hot Encoding and Label Encoder. SO use it.
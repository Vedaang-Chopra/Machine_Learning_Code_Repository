# Introduction to Numpy...................................

# Python Lists are hetrogenous in nature, thus don't have fixed memory and are not compact in size.
# Python lists are slower to process, due to slow nature of python.
# Numpy is homogeneous numbers, and faster to process as it is coded in C.

# Creating Numpy Arrays........................
import numpy as np
l=[1,2,3]
l1=[1,2,'3']
# The array function is used to create an numpy array from a list.
print(np.array(l))          # Numpy array of integers
print(type(np.array(l)))
print(np.array(l1))         # Numpy array of strings due to presence of one string literal

# The zeros and ones function creates an numpy array of float integers of zeros and ones respectively of size mentioned as parameter.
print(np.zeros(10))
print(np.ones(20,'int'))            # To make integer numpy array we can pass it as parameter

# To identify the datatype, just pass the dtype parameter.........
print(np.ones(20,'int').dtype)

# To create a 2-D or 3-D array pass parameters in a tuple
print(np.ones((2,3),'int'))
print(np.ones((2,3,4),'int'))

# Once a numpy is created if a number is passed as string it converts it to string.
a=np.array(l)
a[0]='12'           # a[0] is 12 in integer
print(a,a.dtype)    # Dtype ensures that numpy element are stored as signed 64 bit integers

# Numpy can be accessed similarly as a list.
# e.g.-
print(a[0])         # Accessing Data
print(a[:1])        # Slicing Data
print(a.shape)      # Shape returns Dimensions of Numpy array is returned, Tuple with Size is returned
print(a.size)       # Size Returns the Number of elements present in array
b=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],'int')
print(b[1:3,2])     # Accessing Sub Array


# Timing Numpy .....................
import time
x=np.random.random(1000000)

start=time.time()
sum(x)/len(x)
print('Normal Computation:',time.time()-start)

start=time.time()
x.mean()
print('Numpy Computation',time.time()-start)




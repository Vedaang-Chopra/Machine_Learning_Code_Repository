# Operations on Numpy Arrays........................
import numpy as np
import  math
a=np.array([1,2,3])
b=np.array([4,5,6])

# Numpy_array + constant= Each element is added with the constant
# Numpy_array - constant= Each element is subtracted with the constant
# Numpy_array * constant= Each element is multiplied with the constant
# Numpy_array / constant= Each element is divided with the constant
# Numpy_array ^ constant= Each element is powered with the constant


print(a,'+2',a+2)
print(a,'-2',a-2)
print(a,'*2',a*2)
print(a,'/2',a/2)
print(a,'^2',a**2)

# Note:- The shape of Numpy Array should be same for both.
# Numpy_array_1 + Numpy_array_2= Each element is added corresponding to each other,similar to matrix addition
# Numpy_array_1 - Numpy_array_2= Each element is subtracted corresponding to each other,similar to matrix subtraction
# Numpy_array_1 * Numpy_array_2= Each element is multiplied corresponding to each other, NOT similar to matrix multiplication
# Numpy_array_1 / Numpy_array_2= Each element is divided corresponding to each other, NOT similar to matrix division
# Numpy_array_1 ^ Numpy_array_2= Each element is powered corresponding to each other, NOT similar to matrix pwered

print(a,'+',b,a+b)
print(a,'-',b,a-b)
print(a,'*',b,a*b)
print(a,'/',b,a/b)
print(a,'**',b,a**b)



# Other Numpy Operations.................
print(a.mean())                                   # Mean of numpy elements
print(a.sum())                                    # Sum of numpy elements
print((math.sqrt(((b-a)**2).sum())))              # Root mean Square calculated, assume elements to be cartesian points
# Similarly we can calculated other maths scientific, and statstics problem; vector operations
# e.g.
x=np.array([1,2,3,4,5,6])
print(((x-x.mean())**2).sum())                     # Statistics Standard Deviation
y=np.array([1,2,3,4,5,6])
print(x.dot(y))                                    # Calculate the Dot Product of two elements


# Numpy also has other Data Type such as matrices.
# Numpy matrix are similar to 2-D numpy arrays, but have a little change in behaviour
a1=np.matrix([[1,2,3],[4,5,6]])
a2=np.matrix([[1,2,3],[4,5,6]])
print(a1*a2)                                        # Here Matrix Multiplication Occurs.


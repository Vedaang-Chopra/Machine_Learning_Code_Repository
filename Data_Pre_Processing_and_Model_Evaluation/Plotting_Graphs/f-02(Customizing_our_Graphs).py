import matplotlib.pyplot as plt         # Importing important functions for plotting

# Plotting Non Linear Function and Customizing our Graph..
import numpy as np
x=np.arange(1,5,0.1)
y1=x**1
y2=x**2
y3=x**3
y4=x**4

# color: Parameter to specify colour
# marker: Parameter to mark points with the charachter given
# linewidth: Increase the width of the line,Overshadows the marker parameter
# label:- To specify what each graph represents when there our multiple graphs on a same plot, but only work when called with the legend function.
plt.plot(x,y1,color='green',marker='o',linewidth=1,label='Linear')
plt.plot(x,y2,color='blue',marker='+',label='Quadratic')
plt.plot(x,y3,color='red',marker='*',label='Cubic')
plt.plot(x,y4,color='black',marker='.',label='Biquadratic')

plt.ylabel("Y values for Linear, Quadratic, Cubic and Biquadratic Function")          # To specify what y line represents
plt.xlabel("X values for Linear, Quadratic, Cubic and Biquadratic Function")           # To specify what x line represents
plt.title('Graphs Comparison: Linear v/s Quadratic v/s Cubic v/s Biquadratic')         # Give graph a Title
plt.legend()        # Used to Display the labels of each graph provided with the plot function as parameter

# To change the axis of the graph, what values will the x and y values of the graph will go to we use the axis function.
plt.axis([1,5,0,300])     #([starting x axis point, ending x axis point, starting y axis point, ending y axis point]

# To create a grid on the map use the grid function
plt.grid()


# To add random text at some place on the graph use the title function.
# It requires x,y and text. Also takes fontsize as parameters.
plt.text(1.5,200,'Cubic, Quadratic and Linear Functions')
plt.text(1.5,150,'are Close to each other')

plt.show()















# tid=pd.read_csv('iris_dataframe.csv')
# ti=tid.copy()
# #print(ti.head(20))
# print(ti.shape)
# print(ti.iloc[0:10,0:6])
# x1=[4,5,6]
# y1=[7,8,9]
# print(plt.plot(x,y))
# plt.plot(x1,y1)
# plt.show()
# plt.plot(ti['Age'],ti['Survived'],'bo')
# plt.show()
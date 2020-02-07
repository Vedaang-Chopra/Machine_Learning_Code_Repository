# The Matplot Library is used to Plot Graphs for the data entered to it. The visual representation of the data helps to understand the data more.
# It helps to find patterns in the data.

import matplotlib.pyplot as plt         # Importing important functions for plotting

# As for plotting a 2-D graph we require and x and y point coordinates to plot the graph, Basic requirement for plotting any graph is the x and y coordinates of points.
x=[1,2,3]           # Points coordinates of x-axis for the function assumed
y=[2,4,6]           # Points coordinates of y-axis for the function assumed
# Thus points are (1,2), (2,4) and (3,6) to be plotted.

# The scatter function is used to plot ONLY the points of the graph.It scatters the points all over the graph. It doesn't displays the graph on the console.
plt.scatter(x,y)

# The show function actually prints the graph on the console. It is to be called in the end so that after making all the neccessary adjustements to graph we can print it.
plt.show()
# The colour of the graph point is always random else mentioned.

# **********************************************************************************************************************************************************#

x=[1,2,3]           # Points coordinates of x-axis for the function assumed
y=[2,4,6]           # Points coordinates of y-axis for the function assumed

# To plot a graph rather than present individual points we need to use the plot function. It joins the points we assumed and displays a graph and not individual points.
plt.plot(x,y,'g')

# Note:- 1. Use Plot and scatter simultaneously to see the graph and individual points.
#        2. To change the colour of the graph we can have different colours, and its first charachter would be assumed as its colour.
#             e.g:- For the colour red use r, for green use g etc.
#        3. To change the way of graph use say we want a dashed line/ or points instead of circles nee to be traingles we can use different symbols/special charachters.

plt.scatter(x,y)
plt.show()

# **********************************************************************************************************************************************************#
# To simultaneously plot two graphs on the same graph call plot on on the same plt for different coordinates.
x1=[4,5,6]
y1=[7,8,9]

plt.plot(x,y,'r-')        # Coordinates of First Graph
plt.plot(x1,y1,'b--')     # Coordinates of Second Graph
plt.show()
# **********************************************************************************************************************************************************#
# Plotting Non Linear Function..
import numpy as np
x2=np.array([1,2,3])
y2=x2**3

plt.plot(x2,y2,'g')     # This prints cubic function as 3 lines rather than a cubic function.
plt.show()

# To show a curve rather joined lines, change the x values to be continous rather than discrete.
x2=np.arange(0,5,0.1)
y2=x2**3
plt.plot(x2,y2,'g')     # This prints the cubic function as curve rather than 3 lines rather than a cubic function.
plt.show()
# **********************************************************************************************************************************************************#
# Plotting any list..
a=[1,2,3]
# When we pass a single parameter, then that list turns y and x turns about to be a list that is [i for i in range(0,len(a)-1)]
plt.plot(a)
plt.show()
a=[1,8,27]
plt.plot(a)         # This plots a cubic function due to our input.
plt.show()



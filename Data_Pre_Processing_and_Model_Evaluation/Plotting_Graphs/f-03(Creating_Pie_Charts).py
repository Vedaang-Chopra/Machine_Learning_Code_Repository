import matplotlib.pyplot as plt         # Importing important functions for plotting

# Creating Pie Graphs....................

# Let's assume we have 4 classes and their respective sizes
sizes=[1,2,3.5,4]
# To make the pie graph exactly circle/round use the axis function with equal parameter
plt.axis('equal')

# To change the colours of the pie graph, we can as many of colours as the classes and pass it in the pie function
colours=['black','blue','red','green']

# To put labels on the graph create a labels array and pass it in the pie function
labels=['Class 1','Class 2','Class 3','Class 4']

# To put a title on the graph use title function
plt.title('Plotting a Pie Graph ')

# To display the percentages of the graph use the autopct parameter
# The autopct parameter can take a function which returns a string to be displayed which is the percentage, or use a regular expression.
# We have used the regular expression where % is special character to display % and .2f represents floating precision

# The explode feature helps to break the graph from the starting and display it as individual pie pieces.
# The magnitude with which is separated is to be passed as a list for all classes.
# Then it is to be passed as a parameter to the the pie function.
explode=[0.1,0,0.1,0]

# To make all the parameters that are passed in array to be used be in clockwised direction set the counterclock parameter to be false.
# Default the pie graph is in counter-clockwise direction

# To set the start angle of the graph use the startangle parameter.
plt.pie(sizes,colors=colours,labels=labels,autopct='%.2f%%',counterclock=False,startangle=0,explode=explode)

plt.show()


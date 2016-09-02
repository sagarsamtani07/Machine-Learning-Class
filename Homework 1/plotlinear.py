# plotlinear.py
"""
Homework 1: Sagar Samtani
This script generates linear plots based on various parameters. It is based on Python 2.7.
The first part of the script is the code provided by the instructor.
The second part of the script answers question 11a. I generate three different lines and plot them on a graph and label them accordingly.
The third part of the script was provided by the instructor. To ensure that the code runs smoothly, I have commented this part out.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define two points for the x-axis
x = np.array([-5, 5])

# Define the different intercepts and gradients to plot
w0 = np.arange(0, 20)
w1 = np.arange(0, 8, 0.4)

# Plot all of the lines
plt.figure()
plt.plot()

for i in range(w0.shape[0]):
    plt.plot(x, w0[i] + w1[i]*x)
    print "\ny = " + str(w0[i]) + " + " + str(w1[i]) + " x"

print "\nClose the current plot window to continue"

plt.show()

#############Question 11a: Plot three lines (parameters of your choosing)#############
# defining the two points for the x-axis
xAxis = np.array([-5, 5])

# Plot all of the lines
fig = plt.figure()
plt.plot(xAxis, 3 + 2*x, color = 'red')
red_line = mpatches.Patch(color='red', label='3+2*x: Intercept 3, slope 2')

plt.plot(xAxis, 3 - 2*x, color = 'blue')
blue_line = mpatches.Patch(color = 'blue', label = '3-2*x: Intercept 3, slope -2')

plt.plot(xAxis, 10 - 2*x, color = 'green')
green_line = mpatches.Patch(color = 'green', label = '10-2*x: Intercept 10, slope -2')

fig.suptitle('Plotting three lines with varying parameters', fontsize = 20)
plt.xlabel('x axis', fontsize = 18)
plt.ylabel('y axis', fontsize = 18)

plt.legend(handles=[red_line, blue_line, green_line])

plt.show()




# For the purposes of question 11a, I have commented out the remainder of the script to ensure that I can create my lines without any errors.
# Request user input
# plt.figure()
# plt.plot()
# plt.ion()
# print "\nThe following will ask you for intercept and slope values"
# print "   (assuming floats) and will keep plotting lines on a new plot "
# print "   until you enter 'save', which will save the plot as a pdf "
# print "   called 'line_plot.pdf'."
# print "(NOTE: you may see a MatplotlibDeprecationWarning -- you can safely ignore this)\n"
# while True:
#     intercept = raw_input("Enter intercept: ")
#     if intercept == 'save':
#         break
#     else:
#         intercept = float(intercept)
#
#     gradient = raw_input("Enter gradient (slope): ")
#     if gradient == 'save':
#         break
#     else:
#         gradient = float(gradient)
#
#     plt.plot(x, intercept + gradient*x)
#     plt.show()
#     plt.pause(.1)
#     print "\ny = " + str(intercept) + " + " + str(gradient) + " x\n"
#
# plt.savefig('line_plot.pdf', format='pdf')

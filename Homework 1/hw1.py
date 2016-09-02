"""
Homework 1: Sagar Samtani
This script answers questions 6 - 11. It is based on Python 2.7.
You should be able to run this script in a single shot. There will be multiple outputs which are given based on what is asked in the question.
I have commented the code for each question so you can see what my logic is in each of those areas.
This script assumes that the 'humu.txt' file is in the same directory as the script.
"""
import numpy
import matplotlib.pyplot as plt
from numpy import linalg

# Question 6: loading in the humu.txt file using the loadtxt function
humu = numpy.loadtxt('humu.txt')

print humu

# Question 6: printing out the type the file is
print type(humu)

# printing out the size of the array within the file
print "The size of humu is ", humu.size

# Identifiying the shape of the array within the file
h, w = humu.shape
print "The shape of the array with method 1", h, w

# another way to identify the shape of the array within the file
h, w = humu.shape[0], humu.shape[1]

print "Shape of the array with method 2", h, w

print "Minimum value in humu is",  (humu.min()), "and the maximum value in humu is", (humu.max())

print "Minimum and maximum values for humu array"
print "Minimum value:", min([array.min() for array in humu])
print "Maximum value:", max([array.max() for array in humu])

maxHumuValue = max([array.max() for array in humu])

# scaling the values for the humu array to be between 0 and 1
scaledHumu = numpy.array([[value/float(maxHumuValue) for value in array] for array in humu])

print "Minimum and maximum values for scaledHumu array"
print min([min(array) for array in scaledHumu])
print max([max(array) for array in scaledHumu])

print "Checking that the scaled array (scaledHumu) has the same dimensions as the original array (humu)"

print humu.shape[0], scaledHumu.shape[0]
print humu.shape[1], scaledHumu.shape[1]

# plots the humu (original array from text file). First checks the colormap scheme, then plots the figure with a gray scale colormap
plt.figure()
print plt.cm.cmapname
img = plt.imshow(humu)
plt.show()

plt.figure()
print plt.cm.cmapname
img = plt.imshow(humu, cmap='gray')
plt.show()

# following code makes a uniformly random array with the same dimensions as humu and plots the figure.
randomHumu = numpy.random.random_sample((366,576))

# plot the figure here
plt.figure()
img = plt.imshow(randomHumu, cmap='gray')
plt.show()

'''
Answer to part of the question:
The humu figure and the random humu figure do not look anything alike.
I had expected this, given the random nature of the way the array was generated.
'''

# write the uniformly random array to a file called 'random.txt' using the numpy savetxt function
# Double checks whether the file is the right size and prints out the chart with the same array again.
numpy.savetxt('random.txt', randomHumu)


randomHumuFile = numpy.loadtxt('random.txt')
h, w = randomHumuFile.shape[0], randomHumuFile.shape[1]
print h,w


plt.figure()
img = plt.imshow(randomHumuFile, cmap='gray')
plt.show(img)

#############Question 7#############
def exercise2(infile, outfile):
    """
    This function is all of question 6 put together in a single function. It will take in a file with an array and
    output a file with a random array in it with the same dimensions as the input file.
    :param infile: The infile is the file which is read for the array. For this particular problem, it assumes the input file is humu.
    However, other txt files with arrays can be used as well.
    :param outfile: Returns a txt file with a random array with the same dimensions as the input array
    :return: outfile with a random array.
    """
    humu = numpy.loadtxt(infile)
    print humu
    print type(humu)
    # printing out the size of the array within the file
    print "The size of humu is ", humu.size

    # Identifiying the shape of the array within the file
    h, w = humu.shape
    print "The shape of the array with method 1", h, w

    # another way to identify the shape of the array within the file
    h, w = humu.shape[0], humu.shape[1]

    print "Shape of the array with method 2", h, w
    print "Minimum value in humu is",  (humu.min()), "and the maximum value in humu is", (humu.max())
    print "Minimum and maximum values for humu array"
    print "Minimum value:", min([array.min() for array in humu])
    print "Maximum value:", max([array.max() for array in humu])

    maxHumuValue = max([array.max() for array in humu])

    # scaling the values for the humu array to be between 0 and 1
    scaledHumu = numpy.array([[value/float(maxHumuValue) for value in array] for array in humu])

    print "Minimum and maximum values for scaledHumu array"
    print min([min(array) for array in scaledHumu])
    print max([max(array) for array in scaledHumu])

    print "Checking that the scaled array (scaledHumu) has the same dimensions as the original array (humu)"

    print humu.shape[0], scaledHumu.shape[0]
    print humu.shape[1], scaledHumu.shape[1]

    # plots the humu (original array from text file). First checks the colormap scheme, then plots the figure with a gray scale colormap
    plt.figure()
    print plt.cm.cmapname
    img = plt.imshow(humu)
    plt.show()

    plt.figure()
    print plt.cm.cmapname
    img = plt.imshow(humu, cmap='gray')
    plt.show()

    # following code makes a uniformly random array with the same dimensions as humu and plots the figure.
    randomHumu = numpy.random.random_sample((366,576))

    # plot the figure here
    plt.figure()
    img = plt.imshow(randomHumu, cmap='gray')
    plt.show()

    # write the uniformly random array to a file called 'random.txt' using the numpy savetxt function
    # Double checks whether the file is the right size and prints out the chart with the same array again.
    numpy.savetxt(outfile, randomHumu)

    randomHumuFile = numpy.loadtxt(outfile)
    h, w = randomHumuFile.shape[0], randomHumuFile.shape[1]
    print h,w

    plt.figure()
    img = plt.imshow(randomHumuFile, cmap='gray')
    plt.show(img)



#############Question 9#############
def exercise9():
    """
    This function simulates rolling a pair of dice 1000 times based on a random seed of 8 and calcualtes the probability of getting double sixes.
    :return: print out of probabilities of getting double sixes for 10 procedures.
    """
    # the for loop runs the estimation procedure twice
    for x in range (2):
        # sets the seed to 8 at the beginning of each estimation procedure
        numpy.random.seed(seed=8)
        # this for loop runs the procedure for a total of 10 runs
        for i in range(10):
            # rolling a pair of dice 1000 times
            dice = numpy.random.randint(6, size=(1000, 2))
            # counting how many of the pair rolls are equal to two sixes
            sixSides = 0
            for roll in dice:
                if sum(roll) == 10:
                    sixSides = sixSides + 1
                # calculates the probability of rolling a pair of sixes for this particular procedure
                probOfSixSides = float(sixSides) / 1000
            print "The probability of getting double sixes for the", i, "procedure is ", probOfSixSides
        if x == 1:
            print "Finished both procedures"
        else:
            print "Finished first procedures. Re-initializing seed and running again."


#############Question 10#############
def exercise10():
    """
    Creates three matrices (a, X, and b) and performs various matrix operations on them.
    :return: values for various matrix operations
    """
    # 10a: Write a short script that initializes the random number generator using numpy.random.seed(seed=5) followed by creating two three-dimensional column vectors using numpy.random.rand
    numpy.random.seed(seed = 5)
    a, b = numpy.random.rand(3,1), numpy.random.rand(3,1)
    print "Vector a is:", "\n", a, "\n"
    print "Vector b is:", "\n", b, "\n"

    # 10b
    # adding both arrays
    print "The addition of both vectors is:", "\n", a + b, "\n"

    # element wise multiplication (i.e., Hadamard product, the entrywise product, or the Schur product)
    print "The element wise multiplication of both vectors is:", "\n", numpy.multiply(a,b), "\n"

    # a transpose b (also called the dot product)
    print "The dot product of a transpose and b is:", "\n", numpy.dot(numpy.transpose(a), b), "\n"

    #10c: Set the random seed to 2 and immediately generate a random 3x3 matrix X.
    # display the value of X
    numpy.random.seed(seed = 2)
    X = numpy.random.rand(3,3)
    print "The value of X is:", "\n", X, "\n"

    # a Transpose X
    print "The value of a transpose X is:", "\n", numpy.dot(numpy.transpose(a), X), "\n"
    # a Transpose Xb
    print "The value of a transpose Xb is:", "\n", numpy.dot(numpy.dot(numpy.transpose(a),X), b), "\n"
    # X inverse
    print "The inverse of X is:", "\n", linalg.inv(X)

#############Question 11#############
def exercise11():
    """
    This function plots a sin function between the 0.0 and 10.0 in steps of 0.01
    :return:  a labeled graph with a sine function from 0.0 - 10.0
    """

    # First, generate a vector whose entries are the values of sin(x) for x in the range [0,10] in steps of 0.1
    numpy.arange(0, 10, 0.1)
    x = numpy.linspace(0,10, num=100)

    # second, plot the figure
    fig = plt.figure()
    plt.plot(x, numpy.sin(x))

    # labeling the graph with chart and axes titles
    fig.suptitle('Sine Function for x from 0.0 to 10.0', fontsize = 20)
    plt.xlabel('x values', fontsize = 18)
    plt.ylabel('sin(x)', fontsize = 18)

    plt.show()

exercise2(infile = 'humu.txt', outfile = 'out.txt')
print exercise9()
print exercise10()
print exercise11()
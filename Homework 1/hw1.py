import numpy
import matplotlib.pyplot as plt
from numpy import linalg

# loading in the humu.txt file using the loadtxt function
humu = numpy.loadtxt('humu.txt')

print humu

# printing out the type the file is
print type(humu)

# printing out the size of the array within the file
print humu.size


# Identifiying the shape of the array within the file
h, w = humu.shape
print "Shape of the array with method 1"
print h, w

# another way to identify the shape of the array within the file
h, w = humu.shape[0], humu.shape[1]

print "Shape of the array with method 2"
print h, w

print "Minimum value in humu is",  (humu.min()), "and the maximum value in humu is", (humu.max())

print "Minimum and maximum values for humu array"
print min([array.min() for array in humu])
print max([array.max() for array in humu])


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
# Double checks whether the file is the right size and
numpy.savetxt('random.txt', randomHumu)


randomHumuFile = numpy.loadtxt('random.txt')
h, w = randomHumuFile.shape[0], randomHumuFile.shape[1]
print h,w


plt.figure()
img = plt.imshow(randomHumuFile, cmap='gray')
plt.show()



#############Question 7#############

def exercise2(infile, outfile):
    randomHumu = numpy.random.random_sample((366, 576))
    returnFile = numpy.savetxt(outfile, randomHumu)
    return returnFile

exercise2(infile = 'humu.txt', outfile = 'out.txt')

#############Question 8#############
#NOT SURE IF WE NEED TO HAVE THIS FUNCTION
# def scale01(arr):
#     maxValue = max([array.max() for array in arr])
#     scaledHumu = numpy.array([[value / float(maxValue) for value in array] for array in arr])
#     return scaledHumu


#############Question 9#############
def exercise9():
    for x in range (2):
        numpy.random.seed(seed=8)
        for i in range(10):
            dice = numpy.random.randint(6, size=(1000, 2))
            sixSides = 0
            for roll in dice:
                if sum(roll) == 10:
                    sixSides = sixSides + 1
                probOfSixSides = float(sixSides) / 1000
            print "The probability of getting double sixes for the", i, "procedure is ", probOfSixSides
        print "Finished first run through. Re-initializing seed and running second time."

#############Question 10#############
def exercise10():
    # 10a: Write a short script that initializes the random number generator using numpy.random.seed(seed=5) followed by creating two three-dimensional column vectors using numpy.random.rand
    numpy.random.seed(seed = 5)
    a, b = numpy.random.rand(3,1), numpy.random.rand(3,1)
    print "Vector a is:", "\n", a, "\n"
    print "Vector b is:", "\n", b, "\n"

    # 10b
    # adding both arrays
    addition = a + b
    print "The addition of both vectors is:", "\n", addition, "\n"

    # element wise multiplication (i.e., Hadamard product, the entrywise product, or the Schur product)
    multiply = numpy.multiply(a,b)
    print "The element wise multiplication of both vectors is:", "\n", multiply, "\n"

    # a transpose b (also called the dot product)
    dotProduct = numpy.dot(numpy.transpose(a),b)
    print "The dot product of a and b is:", "\n", dotProduct, "\n"

    #10c: Set the random seed to 2 and immediately generate a random 3x3 matrix X.
    # display the value of X
    numpy.random.seed(seed = 2)
    X = numpy.random.rand(3,3)
    print "The value of X is:", "\n", X, "\n"

    # a Transpose X
    print "The value of a transpose X is:", "\n", numpy.multiply(numpy.transpose(a), X), "\n"
    # a Transpose Xb
    print "The value of a transpose Xb is:", "\n", numpy.multiply(numpy.transpose(a), numpy.multiply(X,b)), "\n"
    # X inverse
    print "The inverse of X is:", "\n", linalg.inv(X)

#############Question 11#############
def exercise11():
    # First, generate a vector whose entries are the values of sin(x) for x in the range [0,10] in steps of 0.1
    numpy.arange(0, 10, 0.1)

    x = numpy.linspace(0,10, num=100)
    fig = plt.figure()
    plt.plot(x, numpy.sin(x))

    # labeling the graph with chart and axes titles
    fig.suptitle('Sine Function for x from 0.0 to 10.0', fontsize = 20)
    plt.xlabel('x values', fontsize = 18)
    plt.ylabel('sin(x)', fontsize = 18)

    plt.show()


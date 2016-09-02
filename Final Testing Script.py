import numpy
import matplotlib.pyplot as plt
from numpy import linalg


def exercise2(infile, outfile):
    """
    This function
    :param infile: The infile is the file which is read for the array. For this particular problem, it assumes the input file is humu.
    However, other txt files with arrays can be used as well.
    :param outfile: Returns a txt file with a random array with the same dimensions as the input array
    :return:
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

#############Question 7#############
exercise2(infile = 'humu.txt', outfile = 'out.txt')
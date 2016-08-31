import numpy
from numpy import linalg
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt
#
# sampleArray = [[2,4,6],
#                [8, 10, 12]]
#
# maxiumum = max([max(array) for array in sampleArray])
#
# print maxiumum
#
# sampleArray2 = [[value/float(maxiumum) for value in array] for array in sampleArray]
#
#
# # for array in sampleArray:
# #     array2 = [value/minimum for value in array]
# #     for value in array:
# #         value = value / minimum
# #         array2.append(value)
# #     sampleArray2.append(array2)
#
# print sampleArray2
#
#
# # randomHumu = numpy.random.random_sample((366,576))
# #
# # plt.figure()
# # img = plt.imshow(randomHumu, cmap='gray')
# # plt.show()
#
# # numpy.savetxt('random.txt', randomHumu)
#
# randomHumuFile = numpy.loadtxt('random.txt')
# h, w = randomHumuFile.shape[0], randomHumuFile.shape[1]
# print h,w
#
#
# plt.figure()
# img = plt.imshow(randomHumuFile, cmap='gray')
# plt.show()

# Set the random seed to 8 using the seed() function: numpy.random.seed(seed=8).
# Use random numpy.random.randint() to produce 1000 throws of two (6 sided) die (randint is zero-based!)
# Use the result to estimate the probability of double sixes.
# REPORT WHAT YOU DID AND THE RESULT.

# numpy.random.seed(seed = 8)
#
# dice = numpy.random.randint(6, size=(1000,2))
# print dice
#
# sixSides = 0
# for roll in dice:
#     if sum(roll) == 10:
#         sixSides = sixSides + 1
# probOfSixSides = float(sixSides)/1000
#
# print sixSides
# print probOfSixSides


# print exercise9()

# def exercise9():
#     for x in range (2):
#         numpy.random.seed(seed=8)
#         for i in range(10):
#             dice = numpy.random.randint(6, size=(1000, 2))
#             sixSides = 0
#             for roll in dice:
#                 if sum(roll) == 10:
#                     sixSides = sixSides + 1
#                 probOfSixSides = float(sixSides) / 1000
#             print "The probability of getting double sixes for the", i, "procedure is ", probOfSixSides
#         print "Finished first run through. Re-initializing seed and running second time."
#
# print exercise9()

# def exercise10():
#     # 10a: Write a short script that initializes the random number generator using numpy.random.seed(seed=5) followed by creating two three-dimensional column vectors using numpy.random.rand
#     numpy.random.seed(seed = 5)
#     a, b = numpy.random.rand(3,1), numpy.random.rand(3,1)
#     print "Vector a is:", "\n", a, "\n"
#     print "Vector b is:", "\n", b, "\n"
#
#     # 10b
#     # adding both arrays
#     addition = a + b
#     print "The addition of both vectors is:", "\n", addition, "\n"
#
#     # element wise multiplication (i.e., Hadamard product, the entrywise product, or the Schur product)
#     multiply = numpy.multiply(a,b)
#     print "The element wise multiplication of both vectors is:", "\n", multiply, "\n"
#
#     # a transpose b (also called the dot product)
#     dotProduct = numpy.dot(numpy.transpose(a),b)
#     print "The dot product of a and b is:", "\n", dotProduct, "\n"
#
#     #10c: Set the random seed to 2 and immediately generate a random 3x3 matrix X.
#     # display the value of X
#     numpy.random.seed(seed = 2)
#     X = numpy.random.rand(3,3)
#     print "The value of X is:", "\n", X, "\n"
#
#     # a Transpose X
#     print "The value of a transpose X is:", "\n", numpy.multiply(numpy.transpose(a), X), "\n"
#     # a Transpose Xb
#     print "The value of a transpose Xb is:", "\n", numpy.multiply(numpy.transpose(a), numpy.multiply(X,b)), "\n"
#     # X inverse
#     print "The inverse of X is:", "\n", linalg.inv(X)

# print exercise10()

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


print exercise11()
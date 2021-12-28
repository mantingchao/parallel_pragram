import pandas
import numpy
import math
df = pandas.read_excel('ping-pong.xlsx')
print('Keys: ', df.keys(), '\n')
messageSize = df.messageSize
sameNode = df.same
diffNode = df.different
z1 = numpy.polyfit(messageSize, sameNode, 1)  # z = np.polyfit(x, y, 3)
print('z1:\n', z1)
print('latency:\n', z1[1])
print('bandwidth:\n', 1 / z1[0] )
z2 = numpy.polyfit(messageSize, diffNode, 1)  # z = np.polyfit(x, y, 3)
print('z2:\n', z2)
print('latency:\n', z2[1])
print('bandwidth:\n', 1 / z2[0] )
print(math.pow(10,-6))
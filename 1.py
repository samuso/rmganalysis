import re
import math
import matplotlib.pyplot as plt

numberOfRuns = 12
# Change of speed required to be considerred as a change
threshhold = 0.01

slow = "too slow"
correct = "just right"
fast = "too fast"

def calculateSpeed(m):
	return (1000.0/(60.0*m))

targetMinutes = [6,9,8,7,6,6,9,7,7,6,6,3,4]
targetSpeeds = map(calculateSpeed, targetMinutes)

# removes Letters from an inputted string
def removeLetters(inputStr):
	return re.findall(r'[0-9].[0-9]+', inputStr)[0]

# parses the data from inputted file
def parseData (content):
	latitude = re.findall(r'51[.][0-9]+', content)
	longitude = re.findall(r'-0.[0-9]+', content)
	timestamp = re.findall(r'2016-.+', content)
	speed = map(removeLetters, re.findall(r'Speed: [0-9].[0-9]+', content))
	signal = re.findall(r'(too fast|just right|too slow)', content)
	return latitude, longitude, timestamp, speed, signal

# returns difference from 
def deviation(values, fromValue):
	diff = 0
	for value in values:
		diff += (float(value) - fromValue) ** 2
	return math.sqrt(diff)/len(values)

# returns metrics relating speed and signals given
def wasEffective(signal, speed):
	adjusted = 0
	stayedright = 0
	nochange = 0
	misinter = 0
	minorpositivechange = 0
	minornegativechange = 0
	for i in range(1, (len(signal))):
		if signal[i-1] == slow:
			
			if (signal[i] == slow):
				nochange+=1
				if (speed[i-1] < speed[i]):
					minorpositivechange+=1
				else:
					minornegativechange+=1

			if (signal[i] == correct):
				adjusted+=1

			if (signal[i] == fast):
				adjusted+=1

		if signal[i-1] == correct:
			if (signal[i] == slow):
				misinter+=1

			if (signal[i] == correct):
				stayedright+=1

			if (signal[i] == fast):
				misinter+=1 

		if signal[i-1] == fast:
			if (signal[i] == slow):
				adjusted+=1

			if (signal[i] == correct):
				adjusted+=1

			if (signal[i] == fast):
				nochange+=1
				if (speed[i-1] > speed[i]):
					minorpositivechange+=1
				else:
					minornegativechange+=1

	return adjusted, nochange, misinter, stayedright, minorpositivechange, minornegativechange

# returns the difference of the average speed from the target speed
def deltaSpeed (speed1, speed2, targetSpeed):
	deltaspeed1 = 0
	deltaspeed2 = 0
	for speed in speed1:
		deltaspeed1 += (float(speed) - targetSpeed)
	deltaspeed1 /= len(speed1)
	for speed in speed2:
		deltaspeed2 += (float(speed) - targetSpeed)
	deltaspeed2 /= len(speed2)

	return deltaspeed1, deltaspeed2

def groupSpeeds (speed):
	newSpeed = []
	for i in range(0, (len(speed)/3)):
		newSpeed.append((float(speed[i]) + float(speed[i+1]) + float(speed[i+2]))/3)
	return newSpeed
# names of the with and without watch run logs
fileName1 = "runnerVibrateLog.txt"
fileName2 = "runnerVibrateLog2.txt"

# stores all the data from the logs
latitudes = []
longitudes = []
timestamps = []
speeds = []
signals = []

# inputs data from files into arrays
for i in range(0,numberOfRuns):
	directory = "/Users/Samusof/Desktop/Uni_Documents/3_Third_Year/RMG/"+ str(i) + "/"

	with open(directory + fileName1) as f:
		lat, lon, t, speed, sig = parseData(f.read())
		latitudes.append(lat)
		longitudes.append(lon)
		timestamps.append(t)
		speeds.append(speed)
		signals.append(sig)

	with open(directory + fileName2) as f:
		lat, lon, t, speed, sig = parseData(f.read())
		latitudes.append(lat)
		longitudes.append(lon)
		timestamps.append(t)
		speeds.append(speed)
		signals.append(sig)

results = []
for i in range (0, numberOfRuns):
	firstRun = wasEffective(signals[2*i], speeds[2*i])
	secondRun = wasEffective(signals[2*i+1], speeds[2*i+1])
	print firstRun
	print secondRun
	k1 = firstRun[0] + firstRun[1] + firstRun[2]
	k2 = secondRun[0] + secondRun[1] + secondRun[2]

	result = (firstRun[0] * k2 - secondRun[0] * k1, firstRun[1] * k2 - secondRun[1] * k1, firstRun[2] * k2 - secondRun[2] * k1)
	results.append(result)
	print results[i]
	print ""

print "\n"

def myError(n1, n2):
	if n1 > n2:
		return 1
	elif n2 > n1:
		return -1
	else:
		return 0

for i in range (0, numberOfRuns):
	deltas = deltaSpeed(speeds[2 * i], speeds[(2 * i) + 1], targetSpeeds[i])
	print deltas
	# print myError(math.fabs(deltas[0]), math.fabs(deltas[1]))

print ""

for i in range (0, numberOfRuns):
	print str(deviation(speeds[(2 * i)], targetSpeeds[i])) + " " + str(deviation(speeds[(2 * i) + 1], targetSpeeds[i]))
print ""
# Here the confusion matrix starts
def findChange(speed, target):
	# 2 is increase
	# 1 is stable speed
	# 0 is decrease
	changes = []
	minDiff = target * threshhold
	for i in range(0, len(speed) - 1):
		s1 = float(speed[i])
		s2 = float(speed[i + 1])
		if s1 - s2 > minDiff:
			changes.append(2)
		elif s2 - s1 > minDiff:
			changes.append(0)
		else:
			changes.append(1)

	return changes

def translateSignals(signal):
	translation = []
	for sig in signal:
		if (sig == slow):
			translation.append(0)
		elif (sig == correct):
			translation.append(1)
		else:
			translation.append(2)

	return translation

# put the speed increase/decrease/stable in an array
def confusionMatrix(speed, targetSpeed, signal):

	speedChange = findChange(speed, targetSpeed)
	translatedSignal = translateSignals(signal)
	similarity = []

	goSlowOnSignal = 0
	goFastOnSignal = 0
	goStableOnSignal = 0

	m = [[0,0,0],[0,0,0],[0,0,0]]

	slowSignal = 0
	fastSignal = 0
	stableSignal = 0

	goodCount = 0
	badCount = 0

	for i in range(0, len(speedChange)):
		if (translatedSignal[i] == 0):
			if (speedChange[i] == 0):
				m[0][0] += 1
			elif (speedChange[i] == 1):
				m[0][1] += 1
			elif (speedChange[i] == 2):
				m[0][2] += 1

			slowSignal += 1
		elif (translatedSignal[i] == 1):
			if (speedChange[i] == 0):
				m[1][0] += 1
			elif (speedChange[i] == 1):
				m[1][1] += 1
			elif (speedChange[i] == 2):
				m[1][2] += 1
			stableSignal += 1
		elif (translatedSignal[i] == 2):
			if (speedChange[i] == 0):
				m[2][0] += 1
			elif (speedChange[i] == 1):
				m[2][1] += 1
			elif (speedChange[i] == 2):
				m[2][2] += 1
			fastSignal += 1


		if (speedChange[i] == translatedSignal[i]):
			similarity.append(1)
			goodCount += 1
		else:
			similarity.append(0)
			badCount += 1
	return m


def addMatrices(m1, m2):
	answer = [[0,0,0],[0,0,0],[0,0,0]]
	for i in range(0, len(m1)):
		for j in range(0, len(m1[i])):
			answer[i][j] = m1[i][j] + m2[i][j]
	return answer

def percentify(m):
	sumOfMatrix = 0
	for i in range(0, len(m)):
		for j in range(0, len(m[i])):
			sumOfMatrix += m[i][j]
	for i in range(0, len(m)):
		for j in range(0, len(m[i])):
			m[i][j] = (float(m[i][j]*100) / sumOfMatrix)
	return m


finalResults = [[],[],[]]
threshhold = 0
for k in range (1, 20):
	threshhold += 0.01
	print "threshhold = " + str(threshhold)
	matrixSum = [[0,0,0],[0,0,0],[0,0,0]]

	for i in range(0, len(speeds)/2):
		tempMatrix = confusionMatrix(speeds[i * 2], targetSpeeds[i], signals[i * 2])
		tempMatrix = percentify(tempMatrix)
		matrixSum = addMatrices(matrixSum, tempMatrix)
		# print tempMatrix[0]
		# print tempMatrix[1]
		# print tempMatrix[2]
		# print ""


	matrixSum = percentify(matrixSum)
	# print matrixSum[0]
	# print matrixSum[1]
	# print matrixSum[2]

	finalResults[0].append(matrixSum[0][0] + matrixSum[1][1] + matrixSum[2][2])
	finalResults[1].append(matrixSum[0][1] + matrixSum[1][0] + matrixSum[1][2] + matrixSum[2][1])
	finalResults[2].append(matrixSum[0][2] + matrixSum[2][0])

	# print "correct: " + str(matrixSum[0][0] + matrixSum[1][1] + matrixSum[2][2]) + "%"
	# print "small mistake: " + str(matrixSum[0][1] + matrixSum[1][0] + matrixSum[1][2] + matrixSum[2][1]) + "%"
	# print "major mistake: " + str(matrixSum[0][2] + matrixSum[2][0]) + "%"
	# print ""

changes = []
for i in range(0, len(speeds)/2):
	for j in range(0, len(speeds[2 * i]) - 1):
		speedDiff = float(speeds[2 * i][j + 1]) - float(speeds[2 * i][j])
		if signals[2 * i][j] == slow :
			if speedDiff > 0:
				changes.append(float(speedDiff) / targetSpeeds[i])

		if signals[2 * i][j] == fast :
			if speedDiff < 0:
				changes.append(float(speedDiff) / targetSpeeds[i])

print len(changes)

def frequencyOfOccurance(list, min, max):
	frequency = 0
	for element in list:
		if element > min:
			if element < max:
				frequency += 1
	return frequency

freq = []
for i in range(0,20):
	min = i * 0.2 - 2
	max = (i+1) * 0.2 - 2
	freq.append(frequencyOfOccurance(changes, min, max))

print freq


plt.plot(freq)
plt.ylabel("delta speed")
plt.show()
# best = 0;
# for i in range(len(finalResults[0])):
# 	if (best < finalResults[0][i] - finalResults[1][i] - finalResults[2][i]):
# 		print finalResults[0][i] - finalResults[1][i] - finalResults[2][i]
# 		print i
# 		best = finalResults[0][i] - finalResults[1][i] - finalResults[2][i]
# the best one is 186

# plt.plot(finalResults[0])
# plt.plot(finalResults[1])
# plt.plot(finalResults[2])
# plt.ylabel('threshhold')
# plt.show()

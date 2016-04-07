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

fo = open("with.txt", "wb")
for i in range(0, len(longitudes[6])):
	fo.write(str(longitudes[6][i])+ " " +str(latitudes[6][i])+ "\n")
fo.close()

fo = open("without.txt", "wb")
for i in range(0, len(longitudes[7])):
	fo.write(str(longitudes[7][i])+ " " +str(latitudes[7][i])+ "\n")
fo.close()

results = []
for i in range (0, numberOfRuns):
	firstRun = wasEffective(signals[2*i], speeds[2*i])
	secondRun = wasEffective(signals[2*i+1], speeds[2*i+1])
	# print firstRun
	# print secondRun
	k1 = firstRun[0] + firstRun[1] + firstRun[2]
	k2 = secondRun[0] + secondRun[1] + secondRun[2]

	result = (firstRun[0] * k2 - secondRun[0] * k1, firstRun[1] * k2 - secondRun[1] * k1, firstRun[2] * k2 - secondRun[2] * k1)
	results.append(result)
# 	print results[i]
# 	print ""

# print "\n"

def myError(n1, n2):
	if n1 > n2:
		return 1
	elif n2 > n1:
		return -1
	else:
		return 0

for i in range (0, numberOfRuns):
	deltas = deltaSpeed(speeds[2 * i], speeds[(2 * i) + 1], targetSpeeds[i])
	# print deltas
	# print myError(math.fabs(deltas[0]), math.fabs(deltas[1]))

# print ""

# for i in range (0, numberOfRuns):
# 	print str(deviation(speeds[(2 * i)], targetSpeeds[i])) + " " + str(deviation(speeds[(2 * i) + 1], targetSpeeds[i]))
# print ""


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

	reactionDistribution = [0] * len(speed)

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
				reactionDistribution[i] = 1
			elif (speedChange[i] == 1):
				m[0][1] += 1
			elif (speedChange[i] == 2):
				m[0][2] += 1

			slowSignal += 1
		elif (translatedSignal[i] == 1):
			if (speedChange[i] == 0):
				m[1][0] += 1
			elif (speedChange[i] == 1):
				reactionDistribution[i] = 1
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
				reactionDistribution[i] = 1
				m[2][2] += 1
			fastSignal += 1


		if (speedChange[i] == translatedSignal[i]):
			similarity.append(1)
			goodCount += 1
		else:
			similarity.append(0)
			badCount += 1

	quartileDistribution = [0,0,0,0]
	length = len(reactionDistribution)
	for i in range(0, length):
		if  reactionDistribution[i] == 1:
			if i < length/4:
				quartileDistribution[0] += 1
			elif i < length/2:
				quartileDistribution[1] += 1
			elif i < length*3/4:
				quartileDistribution[2] += 1
			else:
				quartileDistribution[3] += 1

	return m, quartileDistribution


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
halfFinalResults1 = [[],[],[]]
halfFinalResults2 = [[],[],[]]
threshhold = 0
q1Distribution = []
q2Distribution = []
q3Distribution = []
q4Distribution = []
for k in range (0, 300):
	threshhold += 0.001
	matrixSum = [[0,0,0],[0,0,0],[0,0,0]]
	halfSum1 = [[0,0,0],[0,0,0],[0,0,0]]
	halfSum2 = [[0,0,0],[0,0,0],[0,0,0]]
	reactionDistribution = [0, 0, 0, 0]
	for i in range(0, len(speeds)/2):
		halfSpeeds1 = []
		halfSpeeds2 = []
		halfSignals1 = []
		halfSignals2 = []
		for l in range(0, len(speeds[i*2])):
			if l < len(speeds[i*2])/2:
				halfSpeeds1.append(speeds[i*2][l])
				halfSignals1.append(signals[i*2][l])
			else:
				halfSpeeds2.append(speeds[i*2][l])
				halfSignals2.append(signals[i*2][l])
		halfMatrix1, a = confusionMatrix(halfSpeeds1, targetSpeeds[i], halfSignals1)
		halfMatrix2, a = confusionMatrix(halfSpeeds2, targetSpeeds[i], halfSignals2)
		halfSum1 = addMatrices(halfSum1, halfMatrix1)
		halfSum2 = addMatrices(halfSum2, halfMatrix2)

		tempMatrix, l = confusionMatrix(speeds[i * 2], targetSpeeds[i], signals[i * 2])
		reactionDistribution[0] += l[0]
		reactionDistribution[1] += l[1]
		reactionDistribution[2] += l[2]
		reactionDistribution[3] += l[3]

		tempMatrix = percentify(tempMatrix)
		matrixSum = addMatrices(matrixSum, tempMatrix)
		# print tempMatrix[0]
		# print tempMatrix[1]
		# print tempMatrix[2]
		# print ""

	total = reactionDistribution[0] + reactionDistribution [1] + reactionDistribution [2] + reactionDistribution[3]

	q1Distribution.append(float(reactionDistribution[0]) * 100 / total)
	q2Distribution.append(float(reactionDistribution[1]) * 100 / total)
	q3Distribution.append(float(reactionDistribution[2]) * 100 / total)
	q4Distribution.append(float(reactionDistribution[3]) * 100 / total)

	matrixSum = percentify(matrixSum)
	halfSum1 = percentify(halfSum1)
	halfSum2 = percentify(halfSum2)

	# print matrixSum[0]
	# print matrixSum[1]
	# print matrixSum[2]

	if threshhold > 0.00 and threshhold < 0.002:
		print "yolo"
		print matrixSum[0]
		print matrixSum[1]
		print matrixSum[2]

	finalResults[0].append(matrixSum[0][0] + matrixSum[1][1] + matrixSum[2][2])
	finalResults[1].append(matrixSum[0][1] + matrixSum[1][0] + matrixSum[1][2] + matrixSum[2][1])
	finalResults[2].append(matrixSum[0][2] + matrixSum[2][0])

	halfFinalResults1[0].append(halfSum1[0][0] + halfSum1[1][1] + halfSum1[2][2])
	halfFinalResults1[1].append(halfSum1[0][1] + halfSum1[1][0] + halfSum1[1][2] + halfSum1[2][1])
	halfFinalResults1[2].append(halfSum1[0][2] + halfSum1[2][0])

	halfFinalResults2[0].append(halfSum2[0][0] + halfSum2[1][1] + halfSum2[2][2])
	halfFinalResults2[1].append(halfSum2[0][1] + halfSum2[1][0] + halfSum2[1][2] + halfSum2[2][1])
	halfFinalResults2[2].append(halfSum2[0][2] + halfSum2[2][0])


fo = open("quartiles.txt", "wb")
for i in range(0,len(q1Distribution)):
	fo.write(str(q1Distribution[i]) + " " + str(q2Distribution[i]) + " " + str(q3Distribution[i]) + " " + str(q4Distribution[i]) + "\n")
fo.close()

xAxis = []
for i in range(0, 300):
	xAxis.append((i)*0.001)
plt.plot(xAxis, q1Distribution)
plt.plot(xAxis, q2Distribution)
plt.plot(xAxis, q3Distribution)
plt.plot(xAxis, q4Distribution)
plt.xlabel('Threshold')
plt.ylabel('Percentage of Correct Adjustments')
plt.show()

# def applyConstant(list, constant):
# 	newList = []
# 	for element in list:
# 		newList.append(element*constant)

# 	return newList


# print "This part"
# print halfFinalResults1[0]
# print applyConstant(halfFinalResults1[0], 0.5)

# length = len(halfFinalResults1[0])
# xAxis = []
# for i in range(0, length):
# 	xAxis.append((i)*0.001)
# plt.plot(xAxis, applyConstant(halfFinalResults1[0], 1.1))
# plt.plot(xAxis, applyConstant(halfFinalResults1[1], 0.85))
# plt.plot(xAxis, applyConstant(halfFinalResults1[2], 0.85))
# plt.plot(xAxis, applyConstant(halfFinalResults2[0], 0.9))
# plt.plot(xAxis, applyConstant(halfFinalResults2[1], 1.15))
# plt.plot(xAxis, applyConstant(halfFinalResults2[2], 1.15))
# plt.xlabel('Threshold')
# plt.ylabel('Percent')
# plt.show()



	# print "correct: " + str(matrixSum[0][0] + matrixSum[1][1] + matrixSum[2][2]) + "%"
	# print "small mistake: " + str(matrixSum[0][1] + matrixSum[1][0] + matrixSum[1][2] + matrixSum[2][1]) + "%"
	# print "major mistake: " + str(matrixSum[0][2] + matrixSum[2][0]) + "%"
	# print ""

# This part deals with deltaspeed distribution for with and without watch
def frequencyOfOccurance(list, min, max):
	frequency = 0
	for element in list:
		if element > min:
			if element < max:
				frequency += 1
	return frequency

changes = []
for i in range(0, len(speeds)/2):
	for j in range(0, len(speeds[2 * i]) - 1):
		speedDiff = float(speeds[2 * i][j + 1]) - float(speeds[2 * i][j])
		if signals[2 * i][j] == slow :
			if speedDiff > 0:
				if signals[2 * i][j + 1] == correct or signals[2 * i][j + 1] == fast:
					changes.append(float(speedDiff) / targetSpeeds[i])

		if signals[2 * i][j] == fast :
			if speedDiff < 0:
				if signals[2 * i][j + 1] == correct or signals[2 * i][j + 1] == slow:
					changes.append(float(speedDiff) / targetSpeeds[i])


changes1 = []
for i in range(0, len(speeds)/2):
	index = 2 * i + 1
	for j in range(0, len(speeds[index]) - 1):
		speedDiff = float(speeds[index][j + 1]) - float(speeds[index][j])
		if signals[index][j] == slow :
			if speedDiff > 0:
				changes1.append(float(speedDiff) / targetSpeeds[i])

		if signals[index][j] == fast :
			if speedDiff < 0:
				changes1.append(float(speedDiff) / targetSpeeds[i])

freq = []
for i in range(0,20):
	min = i * 0.2 - 2
	max = (i+1) * 0.2 - 2
	freq.append(frequencyOfOccurance(changes, min, max))

freq1 = []
for i in range(0,20):
	min = i * 0.2 - 2
	max = (i+1) * 0.2 - 2
	freq1.append(frequencyOfOccurance(changes1, min, max))

def plotDeltaDistribution(freq, freq1):
	plt.plot(freq)
	plt.plot(freq1)
	plt.ylabel("delta speed")
	plt.show()

def plotThresholdPerception(correct, incorrect, veryIncorrect):
	length = len(correct)
	xAxis = []
	for i in range(0, length):
		xAxis.append((i)*0.001)
	plt.plot(xAxis, correct)
	plt.plot(xAxis, incorrect)
	plt.plot(xAxis, veryIncorrect)
	plt.xlabel('Threshold')
	plt.ylabel('Percent')
	plt.show()


# now we want to see if the sets are significantly different from each other
diffFromTarget = []
diffFromTarget1 = []
for i in range(0, len(speeds)/2):
	index = 2 * i
	target = targetSpeeds[i]
	for j in range(0, len(speeds[index])):
		diffFromTarget.append((float(speeds[index][j]) - target) / target)
	index = 2 * i + 1
	for j in range(0, len(speeds[index])):
		diffFromTarget1.append((float(speeds[index][j]) - target) / target)

diffFromTarget.sort()
diffFromTarget1.sort()

deltaFrequency = []
for i in range(0,20):
	min = i * 0.2 - 2
	max = (i+1) * 0.2 - 2
	deltaFrequency.append(frequencyOfOccurance(diffFromTarget, min, max))

deltaFrequency1 = []
for i in range(0,20):
	min = i * 0.2 - 2
	max = (i+1) * 0.2 - 2
	deltaFrequency1.append(frequencyOfOccurance(diffFromTarget1, min, max))

# print deltaFrequency
# print deltaFrequency1
# plotDeltaDistribution(deltaFrequency, deltaFrequency1)

# deltaspeed distributions for the ttest to see that they are significantly different from eachother and we're not just lucky
allDeltas = []
allDeltas1 = []
allSpeeds = []
allSpeeds1 = []
for i in range(0, len(speeds)/2):
	index = 2 * i
	target = targetSpeeds[i]
	for j in range(0, len(speeds[index]) - 1):
		allSpeeds.append((float(speeds[index][j])/target) - 1)
		if signals[index][j] != correct:
			allDeltas.append((float(speeds[index][j]) - float(speeds[index][j + 1])) / target)
	index = 2 * i + 1
	for j in range(0, len(speeds[index]) - 1):
		allSpeeds1.append((float(speeds[index][j])/target) - 1)
		if signals[index][j] != correct:
			allDeltas1.append((float(speeds[index][j]) - float(speeds[index][j + 1])) / target)

allDeltas.sort()
allDeltas1.sort()
# print len(allDeltas)

allDeltaFrequency = []
for i in range(0,20):
	min = i * 0.1 - 1
	max = (i+1) * 0.1 - 1
	allDeltaFrequency.append(frequencyOfOccurance(allDeltas, min, max))

allDeltaFrequency1 = []
for i in range(0,20):
	min = i * 0.1 - 1
	max = (i+1) * 0.1 - 1
	allDeltaFrequency1.append(frequencyOfOccurance(allDeltas1, min, max))

print allDeltaFrequency
print allDeltaFrequency1
# plotDeltaDistribution(allDeltaFrequency, allDeltaFrequency1)


allFrequency = []
for i in range(0,20):
	min = i * 0.1 - 1
	max = (i+1) * 0.1 - 1
	allFrequency.append(frequencyOfOccurance(allSpeeds, min, max))

allFrequency1 = []
for i in range(0,20):
	min = i * 0.1 - 1
	max = (i+1) * 0.1 - 1
	allFrequency1.append(frequencyOfOccurance(allSpeeds1, min, max))

print allFrequency
print allFrequency1



fo = open("out.txt", "wb")
for element in allDeltas:
	fo.write(str(element)+"\n")
fo.close()

fo1 = open("out1.txt", "wb")
for element in allDeltas1:
	fo1.write(str(element)+"\n")
fo1.close()

fo1 = open("thresholds.txt", "wb")
for i in range(0, len(finalResults[0])):
	fo1.write(str(finalResults[0][i]) + " " + str(finalResults[1][i]) + " " + str(finalResults[2][i]) + " " + "\n")
fo1.close()

# plotThresholdPerception(finalResults[0], finalResults[1], finalResults[2])
# plotDeltaDistribution(freq, freq1)

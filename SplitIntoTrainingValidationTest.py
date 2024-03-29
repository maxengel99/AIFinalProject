import csv
import random

def continuousToDiscrete(row, featureMap):
	currentDistance = int(row[featureMap['DISTANCE']])
	newDistance = ''
	if currentDistance <= 800:
		newDistance = 'short'
	if currentDistance > 800 and currentDistance <= 1500:
		newDistance = 'medium'
	if currentDistance > 1500:
		newDistance = 'long'
	row[featureMap['DISTANCE']] = newDistance
	
	currentOriginMinTemp = int(row[featureMap['ORIGIN_MIN_TEMPERATURE']])
	newOriginMinTemp = ''
	if currentOriginMinTemp <= 32:
		newOriginMinTemp = 'freezing'
	if currentOriginMinTemp > 32:
		newOriginMinTemp = 'not_freezing'
	row[featureMap['ORIGIN_MIN_TEMPERATURE']] = newOriginMinTemp

	currentDestinationMinTemp = int(row[featureMap['DESTINATION_MIN_TEMPERATURE']])
	newDestinationMinTemp = ''
	if currentDestinationMinTemp <= 32:
		newDestinationMinTemp = 'freezing'
	if currentDestinationMinTemp > 32:
		newDestinationMinTemp = 'not_freezing'
	row[featureMap['DESTINATION_MIN_TEMPERATURE']] = newDestinationMinTemp

	currentOriginMaxTemp = int(row[featureMap['ORIGIN_MAX_TEMPERATURE']])
	newOriginMaxTemp = ''
	if currentOriginMaxTemp <= 90:
		newOriginMaxTemp = 'not_hot'
	if currentOriginMaxTemp > 90:
		newOriginMaxTemp = 'hot'
	row[featureMap['ORIGIN_MAX_TEMPERATURE']] = newOriginMaxTemp

	currentDestinationMaxTemp = int(row[featureMap['DESTINATION_MAX_TEMPERATURE']])
	newDestinationMaxTemp = ''
	if currentDestinationMaxTemp <= 90:
		newDestinationMaxTemp = 'not_hot'
	if currentDestinationMaxTemp > 90:
		newDestinationMaxTemp = 'hot'
	row[featureMap['DESTINATION_MAX_TEMPERATURE']] = newDestinationMaxTemp

	currentOriginWind = float(row[featureMap['ORIGIN_AVG_WIND']])
	newOriginWind = ''
	if currentOriginWind <= 15:
		newOriginWind = 'low_wind'
	if currentOriginWind > 15:
		newOriginWind = 'high_wind'
	row[featureMap['ORIGIN_AVG_WIND']] = newOriginWind

	currentDestinationWind = float(row[featureMap['DESTINATION_AVG_WIND']])
	newDestinationWind = ''
	if currentDestinationWind <= 15:
		newDestinationWind = 'low_wind'
	if currentDestinationWind > 15:
		newDestinationWind = 'high_wind'
	row[featureMap['DESTINATION_AVG_WIND']] = newDestinationWind

	currentOriginSnow = float(row[featureMap['ORIGIN_SNOW_CM']])
	newOriginSnow = ''
	if currentOriginSnow <=.5:
		newOriginSnow = 'no_snow'
	if currentOriginSnow > .5:
		newOriginSnow = 'snow'
	row[featureMap['ORIGIN_SNOW_CM']] = newOriginSnow

	currentDestinationSnow = float(row[featureMap['DESTINATION_SNOW_CM']])
	newDestinationSnow = ''
	if currentDestinationSnow <= .5:
		newDestinationSnow = 'no_snow'
	if currentDestinationSnow > .5:
		newDestinationSnow = 'snow'
	row[featureMap['DESTINATION_SNOW_CM']] = newDestinationSnow

	currentOriginVisibility = float(row[featureMap['ORIGIN_AVG_VISIBILITY']])
	newOriginVisibility = ''
	if currentOriginVisibility <= 2:
		newOriginVisibility = 'low_visibility'
	if currentOriginVisibility > 2 and currentOriginVisibility <= 7:
		newOriginVisibility = 'medium_visibility'
	if currentOriginVisibility > 7:
		newOriginVisibility = 'high_visibility'
	row[featureMap['ORIGIN_AVG_VISIBILITY']] = newOriginVisibility

	currentDestinationVisibility = float(row[featureMap['DESTINATION_AVG_VISIBILITY']])
	newDestinationVisibility = ''
	if currentDestinationVisibility <= 1:
		newDestinationVisibility = 'low_visibility'
	if currentDestinationVisibility > 1 and currentDestinationVisibility <= 5:
		newDestinationVisibility = 'medium_visibility'
	if currentDestinationVisibility > 5:
		newDestinationVisibility = 'high_visibility'
	row[featureMap['DESTINATION_AVG_VISIBILITY']] = newDestinationVisibility

	currentDepartureTime = int(row[featureMap['SCHEDULED_DEPARTURE']]) if row[featureMap['SCHEDULED_DEPARTURE']] != '' else '1200'
	newDepartureTime = ''
	if currentDepartureTime < 1200:
		newDepartureTime = 'morning'
	if currentDepartureTime >= 1200 and currentDepartureTime < 1800:
		newDepartureTime = 'afternoon'
	if currentDepartureTime >= 1800:
		newDepartureTime = 'evening'
	row[featureMap['SCHEDULED_DEPARTURE']] = newDepartureTime

	currentDepartureDelay = int(row[featureMap['DEPARTURE_DELAY']]) if row[featureMap['DEPARTURE_DELAY']] != '' else 0
	newDepartureDealy = ''
	if currentDepartureDelay <= 15:
		newDepartureDealy = 'on_time'
	if currentDepartureDelay > 15:
		newDepartureDealy = 'delayed'
	row[featureMap['DEPARTURE_DELAY']] = newDepartureDealy



with open('Data/added_weather_fields.csv', 'r') as csvFile, open('Data/Training_Data.csv', mode = 'w') as training_file, open('Data/Validation_Data.csv', mode='w') as validation_file, open('Data/Test_Data.csv', mode='w') as test_file, open('Data/Neural_Net_Data.csv', mode='w') as NN_File:
	reader = csv.reader(csvFile)
	training_writer = csv.writer(training_file)
	validation_writer = csv.writer(validation_file)
	test_writer = csv.writer(test_file)
	#NN_wrtier = csv.writer(NN_File)

	headers = reader.next()
	training_writer.writerow(headers)
	validation_writer.writerow(headers)
	test_writer.writerow(headers)
	#NN_wrtier.writerow(headers)

	dataset = []

	for row in reader:
		dataset.append(row)

	random.shuffle(dataset)

	featureMap = {}
	for idx,val in enumerate(headers):
		featureMap[val] = idx

	prunedDelayedDataset = []
	prunedOnTimeDataset = []

	print(sorted(featureMap.keys()))

	cutCounter = 0
	for idx,instance in enumerate(dataset):
		continuousToDiscrete(instance,featureMap)
		delay = instance[featureMap['DEPARTURE_DELAY']]
		if delay == '':
			delay = 0
		if delay == 'delayed':
			prunedDelayedDataset.append(instance)
		elif cutCounter < 200:
			prunedOnTimeDataset.append(instance)
			cutCounter += 1

	random.shuffle(prunedDelayedDataset)
	random.shuffle(prunedOnTimeDataset)

	for idx,row in enumerate(prunedDelayedDataset):
		# NN_wrtier.writerow(row)
		# NN_wrtier.writerow(prunedOnTimeDataset[idx])
		if (idx <= .8 * len(prunedDelayedDataset)):
			training_writer.writerow(row)
			training_writer.writerow(prunedOnTimeDataset[idx])
		elif (idx <= .9 * len(prunedDelayedDataset)):
			validation_writer.writerow(row)
			validation_writer.writerow(prunedOnTimeDataset[idx])
		else:
			test_writer.writerow(row)
			test_writer.writerow(prunedOnTimeDataset[idx])





# date, airline, origin, destination, distance(mi), min_temp(f), max_temp, avg_wind(mph), snowfall(cm), avg_visibility(mi), delay(classifier)
# ignor, feat, feat, feat, short = (0-500) medium = (501-1500) long = (1501+), freezing = (-50-32) not_freezing = (32+), not_hot = (-50-90) hot = (90.1+), low_wind = (0-20), strong_wind = (20+), no_snow = (0-2.5) snow = (2.5+), low_visibility = (0-1) medium_visibility = (1.1-5) high_visibility = (5.1+), on_time = (0-15) delayed(16+)


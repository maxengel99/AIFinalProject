import csv
import random

def continuousToDiscrete(row, featureMap):
	currentDistance = int(row[featureMap['DISTANCE']])
	newDistance = ''
	if currentDistance <= 500:
		newDistance = 'short'
	if currentDistance > 500 and currentDistance <= 1500:
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
	if currentOriginWind <= 20:
		newOriginWind = 'low_wind'
	if currentOriginWind > 20:
		newOriginWind = 'high_wind'
	row[featureMap['ORIGIN_AVG_WIND']] = newOriginWind

	currentDestinationWind = float(row[featureMap['DESTINATION_AVG_WIND']])
	newDestinationWind = ''
	if currentDestinationWind <= 20:
		newDestinationWind = 'low_wind'
	if currentDestinationWind > 20:
		newDestinationWind = 'high_wind'
	row[featureMap['DESTINATION_AVG_WIND']] = newDestinationWind

	currentOriginSnow = float(row[featureMap['ORIGIN_SNOW_CM']])
	newOriginSnow = ''
	if currentOriginSnow <= 2.5:
		newOriginSnow = 'no_snow'
	if currentOriginSnow > 2.5:
		newOriginSnow = 'snow'
	row[featureMap['ORIGIN_SNOW_CM']] = newOriginSnow

	currentDestinationSnow = float(row[featureMap['DESTINATION_SNOW_CM']])
	newDestinationSnow = ''
	if currentDestinationSnow <= 2.5:
		newDestinationSnow = 'no_snow'
	if currentDestinationSnow > 2.5:
		newDestinationSnow = 'snow'
	row[featureMap['DESTINATION_SNOW_CM']] = newDestinationSnow

	currentOriginVisibility = float(row[featureMap['ORIGIN_AVG_VISIBILITY']])
	newOriginVisibility = ''
	if currentOriginVisibility <= 1:
		newOriginVisibility = 'low_visibility'
	if currentOriginVisibility > 1 and currentOriginVisibility <= 5:
		newOriginVisibility = 'medium_visibility'
	if currentOriginVisibility > 5:
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

	currentDepartureDelay = int(row[featureMap['DEPARTURE_DELAY']]) if row[featureMap['DEPARTURE_DELAY']] != '' else 0
	newDepartureDealy = ''
	if currentDepartureDelay <= 15:
		newDepartureDealy = 'on_time'
	if currentDepartureDelay > 15:
		newDepartureDealy = 'delayed'
	row[featureMap['DEPARTURE_DELAY']] = newDepartureDealy



with open('Data/added_weather_fields.csv', 'r') as csvFile, open('Data/Training_Data.csv', mode = 'w') as training_file, open('Data/Validation_Data.csv', mode='w') as validation_file, open('Data/Test_Data.csv', mode='w') as test_file:
	reader = csv.reader(csvFile)
	training_writer = csv.writer(training_file)
	validation_writer = csv.writer(validation_file)
	test_writer = csv.writer(test_file)

	headers = reader.next()
	training_writer.writerow(headers)
	validation_writer.writerow(headers)
	test_writer.writerow(headers)

	dataset = []

	for row in reader:
		dataset.append(row)

	random.shuffle(dataset)

	featureMap = {}
	for idx,val in enumerate(headers):
		featureMap[val] = idx

	prunedDataset = []

	cutCounter = 0
	for idx,instance in enumerate(dataset):
		continuousToDiscrete(instance,featureMap)
		if instance[featureMap['DEPARTURE_DELAY']] == 'delayed':
			prunedDataset.append(instance)
		elif cutCounter < 200:
			prunedDataset.append(instance)
			cutCounter += 1

	random.shuffle(prunedDataset)

	for idx,row in enumerate(prunedDataset):
		if (idx <= .8 * len(prunedDataset)):
			training_writer.writerow(row)
		elif (idx <= .9 * len(prunedDataset)):
			validation_writer.writerow(row)
		else:
			test_writer.writerow(row)





# date, airline, origin, destination, distance(mi), min_temp(f), max_temp, avg_wind(mph), snowfall(cm), avg_visibility(mi), delay(classifier)
# ignor, feat, feat, feat, short = (0-500) medium = (501-1500) long = (1501+), freezing = (-50-32) not_freezing = (32+), not_hot = (-50-90) hot = (90.1+), low_wind = (0-20), strong_wind = (20+), no_snow = (0-2.5) snow = (2.5+), low_visibility = (0-1) medium_visibility = (1.1-5) high_visibility = (5.1+), on_time = (0-15) delayed(16+)


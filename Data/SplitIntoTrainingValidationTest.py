import csv

with open('Data/added_weather_fields.csv') as csvFile:
	reader = csv.DictReader(csvFile)
	for row in reader:
		print(row[year])

from sklearn.naive_bayes import GaussianNB
import csv


TRAINING_FILE = 'Data/Training_Data.csv'
VAILDATION_FILE = 'Data/Validation_Data.csv'
TEST_FILE = 'Data/Test_Data.csv'

def load_data(csvfile, features=['AIRLINE', 'DAY_OF_WEEK', 'DEPARTURE_DELAY', 'DESTINATION_MIN_TEMPERATURE', 'DESTINATION_SNOW_CM', 'DISTANCE', 'MONTH', 'ORIGIN_AVG_VISIBILITY', 'ORIGIN_AVG_WIND', 'ORIGIN_MIN_TEMPERATURE', 'ORIGIN_SNOW_CM', 'SCHEDULED_DEPARTURE']):
  inputdataset = []
  unstructuredDataset = []
  with open(csvfile, newline='\n') as csvfile:
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
      unstructuredDataset.append(row)

  featureList = unstructuredDataset[0]


  for instance in unstructuredDataset[1:]:
    structuredData = {}
    for idx, val in enumerate(featureList):
      if val in features:
      	if val == 'DAY_OF_WEEK' or val == 'MONTH':
      		structuredData[val] = int(instance[idx])
      	else:
      		structuredData[val] = instance[idx]
    inputdataset.append(structuredData)
  
  return inputdataset

training_data = load_data(TRAINING_FILE)

train_x = []
train_y = []

print(training_data[0])

for instance in training_data:
	labels = list(instance.values())
	print(labels)
	train_y.append(labels.pop(4))
	train_x.append(labels)

clf = GaussianNB()
clf.fit(train_x,train_y)




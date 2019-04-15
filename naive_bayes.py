from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import csv


TRAINING_FILE = 'Data/Training_Data.csv'
VAILDATION_FILE = 'Data/Validation_Data.csv'
TEST_FILE = 'Data/Test_Data.csv'

def get_feature_values(dataset, feature):
  return list(set([row[feature] for row in dataset]))

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
validation_data = load_data(VAILDATION_FILE)

feature_values = [] #feature values
label_values = [] #target 

for instance in training_data:
	labels = list(instance.values())
	label_values.append(labels.pop(4))
	feature_values.append(labels)

le = preprocessing.LabelEncoder()

month_arr = []
day_of_week_arr = []
airline_values_arr = []
scheduled_departure_arr = []
distance_val_arr = []
origin_min_temp_arr = []
origin_snow_arr = []
origin_avg_wind_arr = []
origin_avg_visibility_arr = []
dest_min_temp_arr = []
dest_snow_arr = []

for instance in feature_values:
  month_arr.append(instance[0])
  day_of_week_arr.append(instance[1])
  airline_values_arr.append(instance[2])
  scheduled_departure_arr.append(instance[3])
  distance_val_arr.append(instance[4])
  origin_min_temp_arr.append(instance[5])
  origin_snow_arr.append(instance[6])
  origin_avg_wind_arr.append(instance[7])
  origin_avg_visibility_arr.append(instance[8])
  dest_min_temp_arr.append(instance[9])
  dest_snow_arr.append(instance[10])

month_encoded = le.fit_transform(month_arr)
day_of_week_encoded = le.fit_transform(day_of_week_arr)
airline_encoded = le.fit_transform(airline_values_arr)
scheduled_departure_encoded = le.fit_transform(scheduled_departure_arr)
distance_encoded = le.fit_transform(distance_val_arr)
origin_min_temp_encoded = le.fit_transform(origin_min_temp_arr)
origin_snow_encoded = le.fit_transform(origin_snow_arr)
origin_avg_wind_encoded = le.fit_transform(origin_avg_wind_arr)
origin_avg_visibility_encoded = le.fit_transform(origin_avg_visibility_arr)
dest_min_temp_encoded = le.fit_transform(dest_min_temp_arr)
dest_snow_encoded = le.fit_transform(dest_snow_arr)

features = list(zip(month_encoded, day_of_week_encoded, airline_encoded, scheduled_departure_encoded, distance_encoded, origin_min_temp_encoded,
 origin_snow_encoded, origin_avg_wind_encoded,origin_avg_visibility_encoded, dest_min_temp_encoded, dest_snow_encoded))

label_encoded = le.fit_transform(label_values)

model = GaussianNB()
model.fit(features, label_encoded)

feature_values = [] #feature values - validation data
label_values = [] #target - vaidation data

for instance in validation_data:
	labels = list(instance.values())
	label_values.append(labels.pop(4))
	feature_values.append(labels)

month_arr = []
day_of_week_arr = []
airline_values_arr = []
scheduled_departure_arr = []
distance_val_arr = []
origin_min_temp_arr = []
origin_snow_arr = []
origin_avg_wind_arr = []
origin_avg_visibility_arr = []
dest_min_temp_arr = []
dest_snow_arr = []

for instance in feature_values:
  month_arr.append(instance[0])
  day_of_week_arr.append(instance[1])
  airline_values_arr.append(instance[2])
  scheduled_departure_arr.append(instance[3])
  distance_val_arr.append(instance[4])
  origin_min_temp_arr.append(instance[5])
  origin_snow_arr.append(instance[6])
  origin_avg_wind_arr.append(instance[7])
  origin_avg_visibility_arr.append(instance[8])
  dest_min_temp_arr.append(instance[9])
  dest_snow_arr.append(instance[10])

month_encoded = le.fit_transform(month_arr)
day_of_week_encoded = le.fit_transform(day_of_week_arr)
airline_encoded = le.fit_transform(airline_values_arr)
scheduled_departure_encoded = le.fit_transform(scheduled_departure_arr)
distance_encoded = le.fit_transform(distance_val_arr)
origin_min_temp_encoded = le.fit_transform(origin_min_temp_arr)
origin_snow_encoded = le.fit_transform(origin_snow_arr)
origin_avg_wind_encoded = le.fit_transform(origin_avg_wind_arr)
origin_avg_visibility_encoded = le.fit_transform(origin_avg_visibility_arr)
dest_min_temp_encoded = le.fit_transform(dest_min_temp_arr)
dest_snow_encoded = le.fit_transform(dest_snow_arr)

validation_values = []
for x, item in enumerate(month_encoded):
  tmp_arr = []
  tmp_arr.append(month_encoded[x])
  tmp_arr.append(day_of_week_encoded[x])
  tmp_arr.append(airline_encoded[x])
  tmp_arr.append(scheduled_departure_encoded[x])
  tmp_arr.append(distance_encoded[x])
  tmp_arr.append(origin_min_temp_encoded[x])
  tmp_arr.append(origin_snow_encoded[x])
  tmp_arr.append(origin_avg_visibility_encoded[x])
  tmp_arr.append(origin_avg_visibility_encoded[x])
  tmp_arr.append(dest_min_temp_encoded[x])
  tmp_arr.append(dest_snow_encoded[x])
  validation_values.append(tmp_arr)


correct = 0
encoded_label_values = le.fit_transform(label_values)
for x, item in enumerate(validation_values):
  if(model.predict([item])[0] == encoded_label_values[x]):
    correct = correct + 1

print(correct / len(validation_values))

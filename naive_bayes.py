from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import csv

TRAINING_FILE = 'Data/Training_Data.csv'
VAILDATION_FILE = 'Data/Validation_Data.csv'
TEST_FILE = 'Data/Test_Data.csv'

def get_feature_values(dataset, feature):
  return list(set([row[feature] for row in dataset]))

def load_data(csvfile, features=['AIRLINE', 'DISTANCE', 'DAY_OF_WEEK', 'MONTH', 'DESTINATION_MIN_TEMPERATURE', 'DESTINATION_SNOW_CM',
'ORIGIN_AVG_VISIBILITY', 'ORIGIN_AVG_WIND', 'ORIGIN_MIN_TEMPERATURE', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']):
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

def create_feature_and_label_arr(data):
  feature_values = [] #feature values
  label_values = [] #target 
  for instance in training_data:
	  labels = list(instance.values())
	  label_values.append(labels.pop(4))
	  feature_values.append(labels)

  return feature_values, label_values

def encode_features(values, le):
  airline_values_arr = []
  month_arr = []
  day_of_week_arr = []
  dest_min_temp_arr = []
  dest_snow_arr = []
  distance_val_arr = []
  origin_avg_visibility_arr = []
  origin_avg_wind_arr = []
  origin_min_temp_arr = []
  origin_snow_arr = []
  scheduled_departure_arr = []
  distance_arr = []

  for instance in values:
    month_arr.append(instance[0])
    day_of_week_arr.append(instance[1])
    airline_values_arr.append(instance[2])
    scheduled_departure_arr.append(instance[3])
    distance_arr.append(instance[4])
    origin_min_temp_arr.append(instance[5])
    origin_avg_wind_arr.append(instance[6])
    origin_avg_visibility_arr.append(instance[7])
    dest_min_temp_arr.append(instance[8])
    dest_snow_arr.append(instance[9])

  month_encoded = le.fit_transform(month_arr)
  day_of_week_encoded = le.fit_transform(day_of_week_arr)
  airline_encoded = le.fit_transform(airline_values_arr)
  scheduled_departure_encoded = le.fit_transform(scheduled_departure_arr)
  distance_encoded = le.fit_transform(distance_val_arr)
  origin_min_temp_encoded = le.fit_transform(origin_min_temp_arr)
  origin_avg_wind_encoded = le.fit_transform(origin_avg_wind_arr)
  origin_avg_visibility_encoded = le.fit_transform(origin_avg_visibility_arr)
  dest_min_temp_encoded = le.fit_transform(dest_min_temp_arr)
  dest_snow_encoded = le.fit_transform(dest_snow_arr)

  features = list(zip(month_encoded, day_of_week_encoded, airline_encoded, scheduled_departure_encoded, distance_encoded, origin_min_temp_encoded,
  origin_avg_wind_encoded,origin_avg_visibility_encoded, dest_min_temp_encoded, dest_snow_encoded))
  return features

'''
Note - tried to refactor this to make it more readable - it changed the accuracy so I didn't use it
def encode_features_and_create_test(values, le):
  features = encode_features(values, le)
  validation_values = []
  for x, item in enumerate(features):
    tmp_arr = []
    tmp_arr.append(item[0])
    tmp_arr.append(item[1])
    tmp_arr.append(item[2])
    tmp_arr.append(item[3])
    tmp_arr.append(item[4])
    tmp_arr.append(item[5])
    tmp_arr.append(features[x][6])
    tmp_arr.append(features[x][7])
    tmp_arr.append(features[x][8])
    tmp_arr.append(features[x][9])
    tmp_arr.append(features[x][10])
    tmp_arr.append(features[x][11])
    tmp_arr.append(features[x][12])
    validation_values.append(tmp_arr)

  return validation_values
'''

training_data = load_data(TRAINING_FILE)
validation_data = load_data(VAILDATION_FILE)

feature_values, label_values = create_feature_and_label_arr(training_data)
le = preprocessing.LabelEncoder()

encoded_features = encode_features(feature_values, le)

encoded_label = le.fit_transform(label_values)

model = GaussianNB()
model.fit(encoded_features, encoded_label)

validation_feature_values, validation_label_values = create_feature_and_label_arr(validation_data)

airline_values_arr = []
month_arr = []
day_of_week_arr = []
dest_min_temp_arr = []
dest_snow_arr = []
distance_val_arr = []
origin_avg_visibility_arr = []
origin_avg_wind_arr = []
origin_snow_arr = []
origin_min_temp_arr = []
scheduled_departure_arr = []
distance_arr = []

for instance in feature_values:
  month_arr.append(instance[0])
  day_of_week_arr.append(instance[1])
  airline_values_arr.append(instance[2])
  scheduled_departure_arr.append(instance[3])
  distance_arr.append(instance[4])
  origin_min_temp_arr.append(instance[5])
  origin_avg_wind_arr.append(instance[6])
  origin_avg_visibility_arr.append(instance[7])
  dest_min_temp_arr.append(instance[8])
  dest_snow_arr.append(instance[9])

month_encoded = le.fit_transform(month_arr)
day_of_week_encoded = le.fit_transform(day_of_week_arr)
airline_encoded = le.fit_transform(airline_values_arr)
scheduled_departure_encoded = le.fit_transform(scheduled_departure_arr)
distance_encoded = le.fit_transform(distance_val_arr)
origin_min_temp_encoded = le.fit_transform(origin_min_temp_arr)
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
  tmp_arr.append(origin_avg_wind_encoded[x])
  tmp_arr.append(origin_avg_visibility_encoded[x])
  tmp_arr.append(dest_min_temp_encoded[x])
  tmp_arr.append(dest_snow_encoded[x])

# validation_values = encode_features_and_create_test(validation_feature_values, le)
correct = 0
encoded_label_values = le.fit_transform(validation_label_values)
for x, item in enumerate(validation_values):
  if(model.predict([item])[0] == encoded_label_values[x]):
    correct = correct + 1

print(correct / len(validation_values))
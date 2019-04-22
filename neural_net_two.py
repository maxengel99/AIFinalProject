from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd

#ADJUST FEATURES WE WANT TO USE UP HERE
df = pd.read_csv('./Data/Neural_Net_Data.csv')
df = df.reindex(columns=['AIRLINE', 'AIRLINE_DELAY', 'AIR_SYSTEM_DELAY', 'AIR_TIME', 'ARRIVAL_DELAY', 'ARRIVAL_TIME', 'CANCELLATION_REASON', 'CANCELLED', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_DELAY', 'DEPARTURE_TIME', 'DESTINATION_AIRPORT', 'DESTINATION_AVG_VISIBILITY', 'DESTINATION_AVG_WIND', 'DESTINATION_MAX_TEMPERATURE', 'DESTINATION_MIN_TEMPERATURE', 'DESTINATION_SNOW_CM', 'DISTANCE', 'DIVERTED', 'ELAPSED_TIME', 'FLIGHT_NUMBER', 'LATE_AIRCRAFT_DELAY', 'MONTH', 'ORIGIN_AIRPORT', 'ORIGIN_AVG_VISIBILITY', 'ORIGIN_AVG_WIND', 'ORIGIN_MAX_TEMPERATURE', 'ORIGIN_MIN_TEMPERATURE', 'ORIGIN_SNOW_CM', 'SCHEDULED_ARRIVAL', 'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'SECURITY_DELAY', 'TAIL_NUMBER', 'TAXI_IN', 'TAXI_OUT', 'WEATHER_DELAY', 'WHEELS_OFF', 'WHEELS_ON', 'YEAR'])
df.columns = df.columns.to_series().apply(lambda x: x.strip())
df = df[['DAY_OF_WEEK', 'MONTH', 'DAY', 'DISTANCE', 'ORIGIN_AVG_VISIBILITY', 'DESTINATION_AVG_VISIBILITY', 'DESTINATION_AVG_WIND', 'ORIGIN_AVG_WIND', 'DESTINATION_SNOW_CM', 'ORIGIN_SNOW_CM', 'DESTINATION_MIN_TEMPERATURE', 'ORIGIN_MIN_TEMPERATURE', 'DEPARTURE_DELAY']].copy()

#df['DEPARTURE_DELAY_BIN'] = df['DEPARTURE_DELAY'].apply(lambda x: 0 if x <= 15 else 1)

f_feats_wout_missing_data = pd.DataFrame()
for j, x in df.iterrows(): #makes sure all rows have a number (some have NaN for values)
    flag =  True
    for y in x:
        if not(isinstance(y, float)):
            flag = False
    if flag:
        if x['DEPARTURE_DELAY'] > 15:
            x['DEPARTURE_DELAY'] = 1
        else:
            x['DEPARTURE_DELAY'] = 0
        f_feats_wout_missing_data = f_feats_wout_missing_data.append(x)
#        
#f_feats_wout_missing_data = f_feats_wout_missing_data[['DAY', 'DAY_OF_WEEK',  'DESTINATION_AVG_VISIBILITY', 'DESTINATION_AVG_WIND',
#        'DESTINATION_MIN_TEMPERATURE', 'DESTINATION_SNOW_CM', 'DISTANCE', 'MONTH', 'ORIGIN_AVG_VISIBILITY', 'ORIGIN_AVG_WIND', 'ORIGIN_MIN_TEMPERATURE', 'ORIGIN_SNOW_CM', 'DEPARTURE_DELAY']]

dataset = f_feats_wout_missing_data.values #changes pandas df to numpy matrix

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
#dataset = numpy.loadtxt("Data/added_weather_fields.csv", delimiter=",", skiprows=1)
print("Dataset Shape: " + str(dataset.shape))
# split into input (X) and output (Y) variables
X = dataset[:,0:12] #NEEDS TO BE ADJUSTED FOR WHICHEVER FEATURES WE USE
Y = dataset[:,12]

print("Number of Samples: " + str(len(X)))
print("Number of Features: " + str(len(X[0])))

from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 42)

print("Training Samples: " + str(len(X_train)))
print("Testing Samples: " + str(len(X_test)))
print(len(Y_test))

# create model
model = Sequential()
model.add(Dense(12, input_dim=12, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, verbose=0)
scores = model.evaluate(X_train, Y_train)
print(str(scores[0])) #loss
print(str(scores[1])) #accuracy

scores = model.evaluate(X_test, Y_test)
print(str(scores[1])) #accuracy
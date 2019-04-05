from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd

#ADJUST FEATURES WE WANT TO USE UP HERE
df = pd.read_csv('./Data/added_weather_fields.csv')[['DAY_OF_WEEK', 'MONTH', 'DAY', 'DISTANCE', 'ORIGIN_AVG_VISIBILITY', 'DESTINATION_AVG_VISIBILITY', 'DESTINATION_AVG_WIND', 'ORIGIN_AVG_WIND', 'DESTINATION_SNOW_CM', 'ORIGIN_SNOW_CM', 'DESTINATION_MIN_TEMPERATURE', 'ORIGIN_MIN_TEMPERATURE', 'DEPARTURE_DELAY']].copy()

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

dataset = f_feats_wout_missing_data.values #changes pandas df to numpy matrix

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
#dataset = numpy.loadtxt("Data/added_weather_fields.csv", delimiter=",", skiprows=1)
print("Dataset Shape: " + str(dataset.shape))
# split into input (X) and output (Y) variables
X = dataset[:,0:36] #NEEDS TO BE ADJUSTED FOR WHICHEVER FEATURES WE USE
Y = dataset[:,36]

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
model.add(Dense(12, input_dim=36, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, verbose=0)
scores = model.evaluate(X_train, Y_train)
print(str(scores[0]))

scores = model.evaluate(X_test, Y_test)
print(str(scores[1]))
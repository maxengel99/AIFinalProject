from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("Data/added_weather_fields.csv", delimiter=",", skiprows=1)
print("Dataset Shape: " + str(dataset.shape))
# split into input (X) and output (Y) variables
X = dataset[:,0:36]
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
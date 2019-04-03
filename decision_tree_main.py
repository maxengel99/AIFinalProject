import csv
import math
import copy
import random
#import pandas as pd
#from sklearn import tree


TRAINING_FILE = 'Data/Training_Data.csv'
VAILDATION_FILE = 'Data/Validation_Data.csv'
TEST_FILE = 'Data/Test_Data.csv'

#df = pd.read_csv(TRAINING_FILE)
#
#df.head()
#
#df_labels = df[['DEPARTURE_DELAY']].copy()
#df_feats = df[['DESTINATION_MIN_TEMPERATURE', 'ORIGIN_AVG_VISIBILITY']].copy()
#
#df_feats['DESTINATION_MIN_TEMPERATURE'] = df_feats['DESTINATION_MIN_TEMPERATURE'].map({'freezing': 1, 'not_freezing': 0})
#df_feats['ORIGIN_AVG_VISIBILITY'] = df_feats['ORIGIN_AVG_VISIBILITY'].map({'high_visibility': 1, 'medium_visibility': 0})
#df_labels['DEPARTURE_DELAY'] = df_labels['DEPARTURE_DELAY'].map({'on_time': 1, 'delayed': 0})
#
#
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(df_feats, df_labels)
#
#df = pd.read_csv(TEST_FILE)
#
#df.head()
#
#df_labels = df[['DEPARTURE_DELAY']].copy()
#df_feats = df[['DESTINATION_MIN_TEMPERATURE', 'ORIGIN_AVG_VISIBILITY']].copy()
#
#df_feats['DESTINATION_MIN_TEMPERATURE'] = df_feats['DESTINATION_MIN_TEMPERATURE'].map({'freezing': 1, 'not_freezing': 0})
#df_feats['ORIGIN_AVG_VISIBILITY'] = df_feats['ORIGIN_AVG_VISIBILITY'].map({'high_visibility': 1, 'medium_visibility': 0})
#df_labels['DEPARTURE_DELAY'] = df_labels['DEPARTURE_DELAY'].map({'on_time': 1, 'delayed': 0})
#
#res = clf.predict(df_feats)
#
#ind = 0
#count_correct = 0
#for x in df_labels['DEPARTURE_DELAY']:
##    print(x)
#    if x == res[ind]:
#        count_correct = count_correct + 1
#    ind = ind +1
#print(count_correct/len(res))


"""
Take in data from csvfile and generate 
a list of dictionaries where each element 
of the list represents an instance
"""
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
        structuredData[val] = instance[idx]
    inputdataset.append(structuredData)
  
  return inputdataset
    
""""
Return all values of a given feature

:param dataset: List of entries
:param feature: feature to retrieve values for
:return: list of feature values for given feature
"""
def get_feature_values(dataset, feature):
  return list(set([row[feature] for row in dataset]))

"""
Divide dataset on a feature

:param dataset: List of entries
:param feature: Feature to split on
:return: Dictionary with feature values as keys, lists of matching rows as elements 
"""
def split_on_feature(dataset, feature):

  featurevalues = get_feature_values(dataset, feature)
  
  splitData = {}
  for featureval in featurevalues:
    splitData[featureval] = []
  
  for instance in dataset:
    featureval = instance[feature]
    del instance[feature]
    splitData[featureval].append(copy.deepcopy(instance))
    
  return splitData

"""
Calculate entropy 

:param dataset: List of entries
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:return: Entropy of dataset
"""
def entropy(dataset, labelfeature, poslabel='Yes', neglabel='No'):
  poscount = 0
  negcount = 0
  for instance in dataset:
    if instance[labelfeature] == poslabel:
      poscount = poscount + 1
    else:
      negcount = negcount + 1
  total = len(dataset)

  if poscount == total or negcount == total:
    return 0

  entropy = (-poscount/total) * math.log(poscount/total, 2) - (negcount/total) * math.log(negcount/total, 2)
  return entropy

"""
Calculate the information gain of splitting the dataset on a feature

:param dataset: List of entries
:param feature: Feature to split on
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:return: Information gain
"""
def infogain(dataset, feature, labelfeature, poslabel='Yes', neglabel='No'):
  
  # get dict of lists
  split = split_on_feature(dataset, feature)

  # inital entropy  
  h = entropy(dataset, labelfeature, poslabel, neglabel)

  # subtract entropy for each value
  for split in split.values():
    h = h - (len(split) / len(dataset)) * entropy(split, labelfeature,  poslabel, neglabel)

  return h


"""
Get feature with greatest information gain

:param dataset: List of entries
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:param fn: function to use to score features, to override information gain
:return: Name of feature in dataset with greatest information gain after split
"""
def get_best_feature(data, labelfeature, poslabel='Yes', neglabel='No', fn = infogain):
  dataset = copy.deepcopy(data)
  features = dataset[0].keys()
  features = [feature for feature in features if feature != labelfeature]

  # Get pairs of all feature names and their info gains 
  infogains = [(feature, fn(dataset, feature, labelfeature, poslabel, neglabel)) for feature in features]

  # Sort by info gain
  infogains.sort(reverse = True, key = lambda x : x[1])
  return infogains[0][0]


"""
Get label present more in dataset

:param dataset: List of entries
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:return: True/False based on if positive or negative is more common
"""
def get_dominant_label(dataset, labelfeature, poslabel='Yes', neglabel='No'):
  labels = [row[labelfeature] for row in dataset]
  return labels.count(poslabel) > labels.count(neglabel)
"""
Run the ID3 decision tree learning algorithm.

Each node in the tree is either a boolean value (for a leaf node) or a tuple between a string (the name of the feature to decide on) and a dictionary with values of the feature as keys and the next nodes as values.

:param dataset: List of entries
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:param ttl: remaining number of iterations allowed, or -1 to not terminate early
:return: Decision tree generated using ID3 algorithm
"""
def id3(dataset, labelfeature, poslabel='Yes', neglabel='No', ttl = -1):

  # check if all elements are positive
  if all([row[labelfeature] == poslabel for row in dataset]):
    return True
  # check if all elements are negative
  elif all([row[labelfeature] == neglabel for row in dataset]):
    return False
  # check if no features remaining
  elif len(dataset[0].keys()) == 1 or ttl == 0:
    return get_dominant_label(dataset, labelfeature, poslabel, neglabel)

  # otherwise, split on best attribute
  else: 
    best_attribute = get_best_feature(dataset, labelfeature, poslabel, neglabel)

    #print('Splitting on {}'.format(best_attribute))

    # Start with default branch
    node = (best_attribute, {None: get_dominant_label(dataset, labelfeature, poslabel, neglabel)})

    # Add branch for each value for feature
    branch_examples = split_on_feature(dataset, best_attribute)
    
    for key, value in branch_examples.items():
      node[1][key] = id3(value, labelfeature, poslabel, neglabel, ttl - 1)

    return node

"""
Use a decision tree to classify an instance

:param instance: Instance to classify
:param tree: Decision tree to use for classification
:return: Label
"""
def classify(instance, tree):
  # Check for leaf node
  if tree == True:
    return True
  if tree == False:
    return False
  
  # Decision node, determine branch to take
  feature = tree[0]
  branches = tree[1]

  # Go down default branch if no matching branch for feature
  if instance[feature] not in branches.keys():
    return classify(instance, branches[None])

  # Go down the correct branch
  return classify(instance, branches[instance[feature]])

"""
Run test on a dataset

:param dataset: Dataset to classify
:param model: Decision tree to use
:param labelmapping: Dict mapping booleans to label representations in instances
:return: Number representing ratio of correctly classified instances to total number
"""
def accuracy_test(dataset, model, labelmapping = {True: 'Yes', False: 'No'}):
  correct = 0
  for instance in dataset:
    predicted = classify(instance, model)
    if labelmapping[predicted] == instance[labelfeature]:
      correct = correct + 1

  return correct * 100.0 / len(dataset) 

"""
Use reduced-error pruning on a decision tree to prevent overfitting.

:param tree: A decision tree
:param validationset: List of entries
:param labelfeature: Element in entries representing label
:param poslabel: Label for postive instances
:param neglabel: Label for negative instances
:return: pruned tree
"""
def reduced_error_prune(tree, trainingset, validationset, labelfeature, poslabel='Yes', neglabel='No', ttl = -1, labelmapping = {True: 'Yes', False: 'No'}):

  # Allow limited iterations and don't prune leaf nodes
  if ttl == 0 or type(tree) == bool:
    return tree

  # Determine dominant class
  dominant_label = get_dominant_label(trainingset, labelfeature, poslabel, neglabel)

  # Prune if dominant label performs better on validation set
  if accuracy_test(validationset, dominant_label, labelmapping) >= accuracy_test(validationset, tree, labelmapping):
    return dominant_label

  # divide dataset for each branch
  split_train = split_on_feature(trainingset, tree[0])
  split_val = split_on_feature(validationset, tree[0])

  # Prune children
  for branch in tree[1].keys():
    # default branch can't have decision node
    if branch is None:
      continue

    # Cases where it isn't possible to compare
    if branch not in split_val.keys() or branch not in split_train.keys():
      continue
    
    # prune with respective subsets of dataset
    tree[1][branch] = reduced_error_prune(tree[1][branch], split_train[branch], split_val[branch], labelfeature, poslabel, neglabel, ttl - 1, labelmapping)
  return tree


"""
Generator for training and validation sets

:param dataset: Dataset to split
"""
def k_fold_split(dataset, k):
  shuffled = copy.deepcopy(dataset)

  # removed as data was shuffled before upload instead
  #random.shuffle(shuffled)

  # determine size of validation set (train set size is dataset size - valsize)
  valsize = len(shuffled) / k
  for i in range(0, k):
    if i > 0:
      training = copy.deepcopy(shuffled[:int(i * valsize)] + shuffled[int((i + 1)* valsize):])
    else:
      training = copy.deepcopy(shuffled[int((i + 1)* valsize):])

    # Remaining instances are validation
    validation = copy.deepcopy(shuffled[int(i * valsize):int((i + 1) * valsize)])

    # Use as generator function
    yield i, training, validation

"""
Generator for training with one element validation sets

:param dataset: Dataset to split
"""
def leave_one_out_split(dataset):
  # determine size of validation set (train set size is dataset size - valsize)
  valsize = 1
  for i in range(0, len(dataset)):
    if i > 0:
      training = copy.deepcopy(dataset[:int(i * valsize)] + dataset[int((i + 1)* valsize):])
    else:
      training = copy.deepcopy(dataset[int((i + 1)* valsize):])

    # Remaining instances are validation
    validation = copy.deepcopy(dataset[int(i * valsize):int((i + 1) * valsize)])

    # Use as generator function
    yield i, training, validation


#load data
trainingSet = load_data(TRAINING_FILE)
validationSet = load_data(VAILDATION_FILE)
testSet = load_data(TEST_FILE)

labelfeature = 'DEPARTURE_DELAY'
labelmapping = {True: 'delayed', False: 'on_time'}


# This commented code was created for running limited iterations of ID3 and pruning

# Default 
model = True

print("Training model:")

maxdepth = 0

for i in range(1, 17):
  maxdepth = maxdepth + 1
  trainingSet = load_data(TRAINING_FILE)

  # Train tree
  newmodel = id3(trainingSet, labelfeature, 'delayed', 'on_time', i)
  
  # Stop training if stalled
  if newmodel == model:
    print('\tTraining has stopped due to no change in tree this iteration')
    break
  else:
    model = newmodel
  
  print('\tTree trained on training dataset and depth {}:'.format(i))

  trainingSet = load_data(TRAINING_FILE)
  
  validationSet = load_data(VAILDATION_FILE)

  # Test on train
  correct = 0

  for instance in trainingSet:
    if labelmapping[classify(instance, model)] == instance[labelfeature]:
      correct = correct + 1
  
  accuracy = 100.0 * correct / len(trainingSet)
  print('\t\tTraining Accuracy: {}'.format(accuracy))


  # Test on validation
  correct = 0

  for instance in validationSet:
    if labelmapping[classify(instance, model)] == instance[labelfeature]:
      correct = correct + 1
  
  accuracy = 100.0 * correct / len(validationSet)
  print('\t\tValidation Accuracy: {}'.format(accuracy))

# save final tree
finalmodel = copy.deepcopy(model)

# Prune
print("Pruning model:")

for i in range(1, maxdepth):
  print('\tPruning to depth {}'.format(i))
  trainingSet = load_data(TRAINING_FILE)
  
  validationSet = load_data(VAILDATION_FILE)

  # Train tree
  model = reduced_error_prune(copy.deepcopy(finalmodel), trainingSet, validationSet, labelfeature, 'delayed', 'on_time', i, labelmapping)
  
  print('\tPruned tree iteration {}:'.format(i))

  trainingSet = load_data(TRAINING_FILE)
  
  validationSet = load_data(VAILDATION_FILE)

  # Test on train
  accuracy =  accuracy_test(trainingSet, model, labelmapping)
  print('\t\tTraining Accuracy: {}'.format(accuracy))

  # Test on validation
  accuracy = accuracy_test(validationSet, model, labelmapping)
  print('\t\tValidation Accuracy: {}'.format(accuracy))

print(model)


# Earlier code used pre-made 90/10 split
"""
fulldata = copy.deepcopy(trainingSet) + copy.deepcopy(validationSet)

k = int(input('How many cross-validation folds?'))
prune = True

print('-----')

best_tree = True
best_tree_acc = 0
pre_pruning_accuracies = []
post_pruning_accuracies = []

if k > 1:
  print('{}-fold cross validation:'.format(k))

  for i, train, validation in k_fold_split(fulldata, k):
    print("\nSplit {}".format(i))
    print("\nTree:")
    model = id3(copy.deepcopy(train), labelfeature, 'democrat', 'republican')
    #print(model)

    accuracy = accuracy_test(validation, model, labelmapping)
    print('\t\tPre-prune Validation Accuracy: {}'.format(accuracy))
    pre_pruning_accuracies.append(accuracy)

    if prune:
      model = reduced_error_prune(model, copy.deepcopy(train), copy.deepcopy(validation), labelfeature, 'democrat', 'republican', -1, labelmapping)
      
      print('\t\tPruned Tree:')
      print(model)

      accuracy = accuracy_test(validation, model, labelmapping)
      print('\t\tPost-prune Validation Accuracy: {}'.format(accuracy))
      post_pruning_accuracies.append(accuracy)

    if accuracy > best_tree_acc:
      best_tree = copy.deepcopy(model)
      best_tree_acc = accuracy

  # Output results
  print("")
  print("Pre pruning validation accuracy:")
  print("\t{}".format(pre_pruning_accuracies))
  print("\tAverage: {}".format(sum(pre_pruning_accuracies) / k))

  if prune:
    print("")
    print("Post pruning validation accuracy:")
    print("\t{}".format(post_pruning_accuracies))
    print("\tAverage: {}".format(sum(post_pruning_accuracies) / k))
else:
  print('Original data split')

  # no folding
  best_tree = id3(copy.deepcopy(trainingSet), labelfeature, 'delayed', 'on_time')


  accuracy = accuracy_test(validationSet, best_tree, labelmapping)
  print('\t\tPre-prune Validation Accuracy: {}'.format(accuracy))

  if prune:
      best_tree = reduced_error_prune(best_tree, copy.deepcopy(trainingSet), copy.deepcopy(validationSet), labelfeature, 'democrat', 'republican', -1, labelmapping)
      
      accuracy = accuracy_test(validationSet, best_tree, labelmapping)
      print('\t\tPost-prune Validation Accuracy: {}'.format(accuracy))
        


print("")
print("Best Tree:")
print(best_tree)


# Run test
print("")
print("Test set accuracy for best tree: {}".format( accuracy_test(testSet, best_tree, labelmapping)))
"""
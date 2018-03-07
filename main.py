import numpy as np
import pandas

ecoli_data = None
ecoli_features = None
ecoli_classes  = None

image_data = None
image_features = None
image_classes  = None

def subtract_points(a,b):
	return a - b

def square_it(a):
	return a * a
def divide_a_by_b(a,b):
	if b == 0:
		return 0
	return a/b

def get_max_of(a, b):
	if a > b:
		return a
	return b
def take_square_root(a):
	return np.sqrt(a)

def euclidean_dist(arr1, arr2):
	total = 0

	for x,y in zip(arr1,arr2):

		sub = subtract_points(float(x),float(y))
		sqr = square_it(sub)
		total = total + sqr
	return take_square_root(total)

def read_ecoli_csv():
	# Load "ecoli" dataset
	global ecoli_data
	# Read csv in using pandas
	ecoli_data = pandas.read_csv('ecoli.data')

def read_image_csv():
	# Load "image" dataset
	global image_data
	# Read csv in using pandas
	image_data = pandas.read_csv('image.data')

def split_ecoli_data():
	# Get the columns of the ecoli dataset
	cols = ecoli_data.columns
	global ecoli_features
	# Get all but the first column, which are features
	ecoli_features = np.asarray(cols.tolist()[1:-1])

	global ecoli_classes
	# Get just the first column, which are the classes
	ecoli_classes = np.asarray(ecoli_data.iloc[:,-1])

	global ecoli_data
	# Now make all of the data useable
	ecoli_data = np.asarray(ecoli_data.iloc[:,1:-1])
def get_most_common_ele(arr):
	counts = {}
	for x in arr:
	
		if x[0] in counts:
			counts[x[0]] += 1
		else:
			counts[x[0]] = 1
	most = sorted(counts, key=counts.get, reverse=True)
	return most[0]

def split_image_data():
	# Get the columns of the image dataset
	cols = image_data.columns
	global image_features
	# Get all but the first column, which are features
	image_features = np.asarray(cols.tolist()[1:])

	global image_classes
	# Get just the first column, which are the classes
	image_classes = np.asarray(image_data.iloc[:,0])

	global image_data
	# Now make all of the data useable
	image_data = np.asarray(image_data.iloc[:,1:])

def five_fold_cross_validation_folds(data, classes):
	# Get all of the classes for the dataset
	single_classes = np.unique(classes)
	# Make an empty array to hold all of the weights
	class_weights = np.ndarray([])
	# Go through each of the classes
	for c in single_classes:
		# Get their corresponding weights
		total = 0
		length = len(classes)
		for i in range(length):
			if classes[i] == c:
				total += 1
		class_weights = np.append(class_weights,float(total)/float(length))
	# Remove the first number of the ndarray bc it is a 1
	class_weights = np.delete(class_weights, 0,0)
	
	# Get the number of cols in the data
	data_len = len(data[0])

	# Create the empty arrays that will be the folds
	fold1_data = np.ndarray(shape=(1,data_len))
	fold1_class = np.ndarray(shape=(1,1))

	fold2_data = np.ndarray(shape=(1,data_len))
	fold2_class = np.ndarray(shape=(1,1))

	fold3_data = np.ndarray(shape=(1,data_len))
	fold3_class = np.ndarray(shape=(1,1))

	fold4_data = np.ndarray(shape=(1,data_len))
	fold4_class = np.ndarray(shape=(1,1))

	fold5_data = np.ndarray(shape=(1,data_len))
	fold5_class = np.ndarray(shape=(1,1))

	# Number of rows in data
	max_length = len(data)
	# For each class and its corresponding weight
	for c,w in zip(single_classes, class_weights):
		# Get the appropriate number for a single fold
		num_per_fold = ((w * max_length) / 5)-1

		# Fill the folds with the the proper amount of data points
		added_so_far = 0
		for i in range(len(classes)):

			if added_so_far >= num_per_fold:
				break
			if classes[i] == c:

				fold1_data = np.vstack((fold1_data, data[i]))
				fold1_class	= np.vstack((fold1_class, [classes[i]]))

				data = np.delete(data, i,0)
				classes = np.delete(classes, i,0)

				added_so_far += 1

		added_so_far = 0
		for i in range(len(classes)):
			if added_so_far >= num_per_fold:
				break
			if classes[i] == c:
				fold2_data = np.vstack((fold2_data, data[i]))
				fold2_class = np.vstack((fold2_class, classes[i]))

				data = np.delete(data, i,0)
				classes = np.delete(classes, i,0)

				added_so_far += 1

		added_so_far = 0
		for i in range(len(classes)):
			if added_so_far >= num_per_fold:
				break
			if classes[i] == c:
				fold3_data = np.vstack((fold3_data, data[i]))
				fold3_class = np.vstack((fold3_class, classes[i]))

				data = np.delete(data, i,0)
				classes = np.delete(classes, i, 0)

				added_so_far += 1

		added_so_far = 0
		for i in range(len(classes)):
			if added_so_far >= num_per_fold:
				break
			if classes[i] == c:
				fold4_data = np.vstack((fold4_data, data[i]))
				fold4_class = np.vstack((fold4_class, classes[i]))

				data = np.delete(data, i,0)
				classes = np.delete(classes, i,0)

				added_so_far += 1

		added_so_far = 0

		for i in xrange(len(classes)):
			if added_so_far >= num_per_fold:
				break
			if classes[i] == c:
				fold5_data = np.vstack((fold5_data, data[i]))
				fold5_class = np.vstack((fold5_class, classes[i]))

				added_so_far += 1

	fold1_data = np.delete(fold1_data, 0,0)
	fold1_class = np.delete(fold1_class, 0,0)
	fold2_data = np.delete(fold2_data, 0,0)
	fold2_class = np.delete(fold2_class, 0,0)
	fold3_data = np.delete(fold3_data, 0,0)
	fold3_class = np.delete(fold3_class, 0,0)
	fold4_data = np.delete(fold4_data, 0,0)
	fold4_class = np.delete(fold4_class, 0,0)
	fold5_data = np.delete(fold5_data, 0,0)
	fold5_class = np.delete(fold5_class, 0,0)

	return [fold1_data, fold1_class, fold2_data, fold2_class, fold3_data, fold3_class, fold4_data, fold4_class, fold5_data, fold4_class ]
def get_neighbors(training, test_point, k):
	import operator
	distances = []
	index = 0
	for neighbor in training:
		dist = euclidean_dist(neighbor, test_point)
		# distances.append((neighbor, dist, index))
		distances.append((index, dist))
		index += 1
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors
def test_run_knn(test_data, test_classes, train_data, train_classes, k):
	num_correct = 0
	num_points = len(test_classes)

	for class_of, point in zip(test_classes, test_data):
		neighbors_indexes= get_neighbors(train_data, point, k)
		neighbors = []
		for i in neighbors_indexes:
			neighbors.append(train_classes[i])
		desicion = get_most_common_ele(neighbors)
		if desicion == class_of:
			num_correct += 1
	return float(num_correct)/float(num_points)

def k_nearest_neighbors(data, classes, k):
	folds = five_fold_cross_validation_folds(data, classes)
	scores = []
	for i in range(0,10,2):
		testing_data = folds[i]
		testing_classes = folds[i+1]
		training_data = None
		training_classes = None
		added_data = False
		added_classes = False
		for x in range(0,10,2):
			if x != i:
				if added_data:
					training_data = np.append(training_data, folds[x], axis=0)
				else:
					training_data = folds[x]
					added_data = True

				if added_classes:
					training_classes = np.append(training_classes, folds[x+1], axis=0)
				else:
					training_classes = folds[x+1]
					added_classes = True
		scores.append(test_run_knn(testing_data, testing_classes, training_data, training_classes, k))
	return np.average(scores)
def tune_knn(data, classes):
	top = 0
	best_so_far = 0
	maxi = len(np.unique(classes))
	for i in range(1, maxi):
		pref = k_nearest_neighbors(data, classes, i)
		if pref > best_so_far:
			top = i
			best_so_far = pref

	print "Best K = " + str(top) + " Perf: " + str(best_so_far)
read_ecoli_csv()
split_ecoli_data()

read_image_csv()
split_image_data()
# folds = five_fold_cross_validation_folds(ecoli_data, ecoli_classes)
# neighbors = get_neighbors(folds[0], folds[2][30],3)
# print folds[3][30]
# for i in neighbors:
# 	print folds[1][i[0]]
tune_knn(image_data, image_classes)

import sys
import numpy as np
import pandas

# global that will be the dataset to use
data_set = None
# global that will be the feature set of the data
feature_set = None
# global that will be the class set of the data
class_set = None


#
# This function will subtract the first parameter from the second parameter
#
def subtract_points(a,b):
	# Subtract the points
	return a - b

#
# This function will square the parameter
#
def square_it(a):
	# Square parameter by multiplying it by itself
	return a * a

#
# This function will divide the first parameter by the second parameter
#
def divide_a_by_b(a,b):
	# Check to make sure that the divisor is not zero
	if b == 0:
		# If the divisor is 0 return 0
		return 0
	# Divid a by b
	return float(a)/float(b)

#
# Return the greater of the two parameters that were passed in
# If they are equal, the second parameter will be returned
#
def get_max_of(a, b):
	# If a is larger than b 
	if a > b:
		# Return a
		return a
	# Otherwise return b
	return b

#
# This function will take the square root of the parameter passed to it
#
def take_square_root(a):
	# Return the square root of a
	return np.sqrt(a)

#
# This function will get the average value of the array that was passed in
#
def get_average(arr):
	# Set a running total to zero
	total = 0
	# Set a running count to zero
	count = 0
	# For each element in the array
	for ele in arr:
		# Add the element to the running total
		total += ele
		# Increment the count by 1
		count += 1
	# Return the total divided by the count
	return divide_a_by_b(float(total),float(count))

#
# This function will return the squared error of from the prediction(param) and the test_point(param)
#
def square_error(prediction, test_point):
	# Get the error by subtracting the two points 
	error = subtract_points(prediction, test_point)
	# Square the error you just calculated and return it
	return  square_it(error)

#
# This function will get the Euclidean Distance between the two arrays passed to it
# Parameters must be arrays where the indexes are of the same axis
#
def euclidean_dist(arr1, arr2):
	# Set a running total to zero
	total = 0
	# Walk along the two arrays and get the corresponding indexes
	for x,y in zip(arr1,arr2):
		# Subtract the two indexes from each other
		sub = subtract_points(float(x),float(y))
		# Square the subtraction of the two indexes
		sqr = square_it(sub)
		# Add to the running total
		total = total + sqr
	# Return the square root of the running total
	return take_square_root(total)

#
# This function will read in a csv which is located at the parameter passed to it
# 
def read_csv(filepath):
	# Get a reference to the global variable data_set
	global data_set
	# Read csv in using pandas and set it to the data_set global
	data_set = pandas.read_csv(filepath)
	# Randomize the data
	print data_set.shape[0]
	data_set = data_set.sample(n=data_set.shape[0])

#
# This function will split the global data_set into features, classes, and measureable data
#
def split_data():
	# Get the columns of the dataset
	cols = data_set.columns
	# Get a reference to the global feature_set
	global feature_set
	# Get all of the features by reading in all but the last column of the first row
	feature_set = np.asarray(cols.tolist()[:-1])
	# Get a reference tp the global class_set
	global class_set
	# Get the entire last column of the dataset
	class_set = np.asarray(data_set.iloc[:,-1])
	# Get a reference to the global data_set
	global data_set
	# Now take all of the columns but the last
	data_set = np.asarray(data_set.iloc[:,:-1])

#
# This function will return the most common element of the array that was passed as a parameter
#
def get_most_common_ele(arr):
	# Set a list that will keep track of elements and their counts
	counts = {}
	# For each element in the array that was passed in
	for ele in arr:
		# If the element is already in the counts list
		if ele[0] in counts:
			# Increment the corresponding element in the counts list by 1
			counts[ele[0]] += 1
		# Otherwise add the element to the counts array
		else:
			counts[ele[0]] = 1
	# Sort the counts list and put the most on the top of the list
	most = sorted(counts, key=counts.get, reverse=True)
	# Return the top element from the sorted list
	return most[0]

#
# This is a function that will take in the data and its corresponding classes and separates them into
# five stratified folds
#
def five_fold_cross_validation_folds(data, classes):
	# Get all of the classes for the dataset
	single_classes = np.unique(classes)
	# Make an empty array to hold all of the weights
	class_weights = np.ndarray([])
	# Go through each of the classes
	for s_class in single_classes:
		# Get their corresponding weights
		# Set a running total
		total = 0
		# Get the number of all of the classes
		length = len(classes)
		# Iterated through all of the classes
		for index in range(length):
			# If you find a class in the classes array
			if classes[index] == s_class:
				# Increment the running total by 1
				total += 1
		# Calculate the weight of the corresponding class
		weight = divide_a_by_b(float(total),float(length))
		# Append weight to running total
		class_weights = np.append(class_weights,weight)
	# Remove the first number of the ndarray bc it is a 1
	class_weights = np.delete(class_weights, 0,0)
	
	# Get the number of cols in the data
	data_len = len(data[0])

	folds = []
	# Create the empty arrays that will be the folds
	for count in range(5):
		# Make an empty array that will be the correct size
		temp = np.ndarray(shape=(1,data_len))
		# temp = np.ones([1,data_len])
		# Add a data fold to the folds array
		folds.append(temp)
		# Make an empty array that will be the correct size
		temp1 = np.ndarray(shape=(1,1))

		# Add a class fold to the folds array
		folds.append(temp1)

	# Number of rows in data
	max_length = len(data)
	
	# For each class and its corresponding weight
	for s_class,class_weight in zip(single_classes, class_weights):
		# Get the appropriate number for a single fold
		num_per_fold = ((class_weight * max_length) / 5)-1

		# Fill the folds with the the proper amount of data points
		# Iterate through folds array 2 by 2
		for index in range(0, 10, 2):
			# Add a running total of how many you have added so far (class instances)
			added_so_far = 0
			# For all of the classes 
			for i in range(len(classes)):
				# If you have added enough class instances to this fold
				if added_so_far >= num_per_fold:
					# Skip
					break
				# If you found a class instance
				if classes[i] == s_class:
					# Add the corresponding data instance to the fold
					folds[index] = np.vstack((folds[index], data[i]))
					# Add the corresponding class instance to the fold
					folds[index + 1] = np.vstack((folds[index + 1], [classes[i]]))
					# If you are still on beginning folds
					if index < 8:
						# Remove the corresponding data instance from the data list
						data = np.delete(data, i,0)
						# Remove the corresponding class instance from the class list
						classes = np.delete(classes, i,0)
						# Increment the running total of instances you have added so far
					added_so_far += 1
	# For each array in the folds array
	for index in range(len(folds)):
		# Remove the first element because it is a placeholder
		folds[index] = np.delete(folds[index], 0,0)
	# Return the folds
	return folds

#
# This function will get k(param) the neighbors of the test_point(param) from the training(param)
#
def get_neighbors(training, test_point, k):
	import operator
	# Set an array that will hold all of the distances
	distances = []
	# Set an index
	index = 0
	# For each neighbor in the training set
	for neighbor in training:
		# Get the euclidean distance of that neighbor to form the test_point
		dist = euclidean_dist(neighbor, test_point)
		# Append the distance and the corresponding index of that instance to the distances array
		distances.append((index, dist))
		# Increment the index counter by 1
		index += 1
	# Sort the distances array
	distances.sort(key=operator.itemgetter(1))
	# Create an array to house all of the neighbors
	neighbors = []
	# From 0 to k
	for i in range(k):
		# Add the next closest neighbor to the array from the sorted distances array
		neighbors.append(distances[i][0])
	# Return the neihbors array
	return neighbors

#
# This function will run the knn algo using regression
#
def test_run_knn_regression(test_data, test_classes, train_data, train_classes, k):
	# Set an array to save all of the errors while you test
	errors = []
	# For each point and its corresponding "class", which is its measurment
	for class_of, point in zip(test_classes, test_data):
		# Get the indexes of its nearest k neighbors
		neighbors_indexes= get_neighbors(train_data, point, k)
		# Set an array to put all of the neighbors
		neighbors = []
		# For each of the indexes of your k nearest neighbors
		for i in neighbors_indexes:
			# Append the corresponding "class", which is its measurment
			neighbors.append(train_classes[i])
		# Set the prediction of what the "class" should be, which is its measurment by averaging k of the neighbors
		prediction = get_average(neighbors)
		# Get the square error of from the prediction and the point
		square_errord = square_error(prediction, class_of)
		# Append the squared error to the "errors" array
		errors.append(square_errord)
	# Return the average of the errors array which is you mean squared error for this run
	return get_average(errors)

#
# This function will run the knn algo using classification
#
def test_run_knn_classification(test_data, test_classes, train_data, train_classes, k):
	# Set the number of correct guesses to zero
	num_correct = 0
	# Get the number of points that you must test
	num_points = len(test_classes)
	# For each point and its corresponding class
	for class_of, point in zip(test_classes, test_data):
		# Get the index of its k nearest neighbors
		neighbors_indexes= get_neighbors(train_data, point, k)
		# Set an array to put all of the neighbors into
		neighbors = []
		# For each of the indexes of your k nearest neighbors
		for i in neighbors_indexes:
			# Append the corresponding class to neighbors array
			neighbors.append(train_classes[i])
		# The class descion guess is the most common class in the neighbors array
		desicion = get_most_common_ele(neighbors)
		# If the desicion you made was correct
		if desicion == class_of:
			# Increment the number of correct guesses by 1
			num_correct += 1
	# Return the number of correct guesses by the total number of points
	return divide_a_by_b(float(num_correct),float(num_points))

#
# This function will create the training and testing sets by combining folds
# param 'i' is the fold number you want to test
#
def create_training_and_test_sets(i, folds):
	# Get a reference to the fold that will be the testing data set
	testing_data = folds[i]
	# Get a reference to the fold that is the testing classes set
	testing_classes = folds[i+1]
	# Set a reference to the training data set
	training_data = None
	# Set a reference to the training class set
	training_classes = None
	# Add a bool that will determine if we have added data to the training data set
	added_data = False
	# Add a bool that will determine if we have added classes to the training class set
	added_classes = False
	# For each of the folds
	for x in range(0,10,2):
		# If you are not on the corresponding fold number to test
		if x != i:
			# If you already added data to the training_data dataset
			if added_data:
				# Append the fold you are on to the training_data dataset
				training_data = np.append(training_data, folds[x], axis=0)
			# Otherwise
			else:
				# Set the training_data dataset to the the fold you are currently on
				training_data = folds[x]
				# Set the added_data bool to be true becasue you just added data
				added_data = True
			# If you have already added classes to the training_classes class set
			if added_classes:
				# Append the class fold that you are currently on
				training_classes = np.append(training_classes, folds[x+1], axis=0)
			# Otherwise
			else:
				# Set the training_class class set to be the fold you are currently on
				training_classes = folds[x+1]
				# Set the added_classes bool to be true because you just added data
				added_classes = True
	# Return all of the data and class sets
	return (testing_data, testing_classes, training_data, training_classes)

#
# This function will prefore the knn algo with either classification or regression
#
def k_nearest_neighbors(data, classes, k, knn_type):
	# Get the five folds to test
	folds = five_fold_cross_validation_folds(data, classes)
	# Set an array to house all of the scores you get from the five test runs
	scores = []
	# For each fold, which you iterate thorugh two by two to account for the data and class folds
	for i in range(0,10,2):
		# Get the test and training sets
		test_train_sets = create_training_and_test_sets(i, folds)
		# If you are doing classification
		if knn_type == "classification":
			# Preform the condensed nearest neighbors on the training set
			cnn = condensed_nearest_neighbor(test_train_sets[0], test_train_sets[1]) 
			# Preform the knn classification test run with the training and test data sets
			scores.append(test_run_knn_classification(cnn[0], cnn[1], test_train_sets[2], test_train_sets[3], k))
		# Otherwise you are doing regression
		else:
			# Preform the knn regresson test run with the training and test data sets
			scores.append(test_run_knn_regression(test_train_sets[0], test_train_sets[1], test_train_sets[2], test_train_sets[3], k))
	# Get the average of the scores, this will be your preformance for this knn run			
	return np.average(scores)

#
# This function will tune your knn algo to find the best k
#
def tune_knn(data, classes, knn_type):
	# Set a running best number for k
	top = 0
	# Set a running best preformance so far
	best_so_far = 0
	# Set a maximum of k so you do not run forever
	maxi = 10
	# For k beginning at 1 and running to the mac k you wish to test 
	for k in range(1, maxi):
		# Get the preformance of the knn run
		pref = k_nearest_neighbors(data, classes, k, knn_type)
		# If you are doing classification for knn
		if knn_type == "classification":
			# If the preformance is greater than the best_so_far performance
			if pref > best_so_far:
				# Set top to be the k
				top = k
				# Set the best_so_far performace to be the one you just calculated
				best_so_far = pref
		# Otherwise you are doing regression
		else:
			# If this is probably your first run or you have some weird data
			# Your best_so_far performance willb be zero
			if best_so_far is 0:
				# Set the top to be k
				top = k
				# Set the best_so_far performace to be the one you just calculated
				best_so_far = pref
			# If your performance is less than best_so_far (which is good for regression)
			elif pref < best_so_far:
				# Set the top to be k
				top = k
				# Set the best_so_far performace to be the one you just calculated
				best_so_far = pref
			# Just pass if you did not make the cut
			else:
				pass
	# Tell the user the best k and its performance
	print "Best K = " + str(top) + " Performance: " + str(best_so_far)

#
# This function will preform the condesed nearest neighbors algo on the data passed in
#
def condensed_nearest_neighbor(training_data, training_classes):
	# Get length of training data
	training_data_length = len(training_data)
	# Start with an empy set Z for points
	z = []
	# Have another set for classes of the points in z
	z_class = []
	# Add an arbitray point to Z
	z.append(training_data[0])
	# Add an arbitraty point to Z for classification
	z_class.append(training_classes[0])
	# Set a value to see if Z has been changed
	change = True
	# Set the length of Z to reference it in while loop
	z_length = len(z)
	
	# Iterate until no change in Z
	while(change):
		# Set a list for random index numbers
		rand_nums = list(range(0,training_data_length-1))
		# Randomize indexes
		np.random.shuffle(rand_nums)
		# Iterate through all of x in X (training set) in random order
		for row_index in rand_nums:
			# Get the point class nearest me
			x_prime = get_neighbors(z, training_data[row_index], 1)
			# Get the class of the only neighbor
			x_prime = z_class[x_prime[0]]
			# If missclassified
			if training_classes[row_index] != x_prime:
				# Add to Z
				z.append(training_data[row_index])
				# Add to Z
				z_class.append(training_classes[row_index])
		# If length of Z is the same as beginning of loop
		if len(z) == z_length:
			# Nothing changed and you can stop loop
			change = False
		else:
			# Set the new length of z
			z_length = len(z)
	# # Get a reference to the global dataset
	# global data_set
	# # Set the global dataset
	# data_set = z
	# # Get a rederence to the global dataset
	# global class_set
	# # Set the global class set
	# class_set = z_class
	return [z, z_class]
# Read in the file passed in by the command line when script started
read_csv(sys.argv[1])
# Split the data to be used
split_data()
# Read in the second argument passed in by the command line and use it as they type of knn
knn_type = sys.argv[2]

print "###### RESULTS ######"
# Tune the data to find the best k
tune_knn(data_set, class_set, str(knn_type))




import numpy as np
import pandas

ecoli_data = None
ecoli_features = None
ecoli_classes  = None

image_data = None
image_features = None
ecoli_classes  = None

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

def split_image_data():
	# Get the columns of the image dataset
	cols = image_data.columns
	global image_features
	# Get all but the first column, which are features
	image_features = np.asarray(cols.tolist()[1:])

	global ecoli_classes
	# Get just the first column, which are the classes
	image_classes = np.asarray(image_data.iloc[:,0])

	global image_data
	# Now make all of the data useable
	image_data = np.asarray(image_data.iloc[:,1:])

def five_fold_cross_validation(data, classes):
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


read_ecoli_csv()
split_ecoli_data()

read_image_csv()
split_image_data()
five_fold_cross_validation(ecoli_data, ecoli_classes)
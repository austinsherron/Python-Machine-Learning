################################################################################
## IMPORTS #####################################################################
################################################################################


from csv import reader


################################################################################
################################################################################
################################################################################


################################################################################
## UTILITY FUNCTIONS ###########################################################
################################################################################


def load_data_from_csv(csv_path, label_index, trans_func=lambda x: x):
	"""
	Function that loads from a CSV into main memory.

	Parameters
	----------
	csv_path : str
		Path to CSV file that contains data.
	label_indes : int
		The index in the CSV rows that contains the label
		for each data point.
	trans_func : function object
		Function that transform values in CSV, i.e.: str -> int.

	Returns
	-------
	data,labels : (list)
		Tuple that contains a list of data points (index 0) and
		a list of labels corresponding to thos data points (index 1).
	"""
	data = []
	labels = []

	with open(csv_path) as f:
		csv_data = reader(f)
	
		for row in csv_data:
			row = list(map(trans_func, row))

			labels.append(row.pop(label_index))
			data.append(row)

	return data,labels


def filter_data(data, labels, filter_func):
	"""
	Function that filters data based on filter_func. Function
	iterates through data and labels and passes the values
	produced by the iterables to filter_func. If filter_func
	returns True, the values aren't included in the return
	arrays.

	Parameters
	----------
	data : array-like
		Array that contains data points.
	labels : array-like
		Array that contains labels.
	filter_func : function object
		Function that filters data/labels.

	Returns
	-------
	filtered_data,filtered_labels : (list)
		Filtered arrays.
	"""
	filtered_data,filtered_labels = [], []
	for point,label in zip(data,labels):
		if not filter_func(point,label):
			filtered_data.append(point)
			filtered_labels.append(label)

	return filtered_data,filtered_labels


################################################################################
################################################################################
################################################################################

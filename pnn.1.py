
import numpy as np
import pandas as pd
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt

# 1. Read the data points
a = pd.read_csv('./app/data/2016.csv',header=0)
b = pd.read_csv('./app/data/2016_2.csv', header=0)
c = pd.read_csv('./app/data/2017.csv', header=0)
d = pd.read_csv('./app/data/2018.csv', header=0)

important = ["CVE_DIAB","PESO","ESTATURA"]

label="CVE_DIAB"


result = pd.concat([a, b, c, d])

result= result.replace("Si", 1)
result= result.replace("No", 0)
result['CVE_DIAB'] = result['CVE_DIAB'].replace(0,2)

result.fillna(0)

result= result[important]

                            

cols_to_norm = [ "PESO"  , "ESTATURA"]

result[cols_to_norm] = result[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#print(result.to_string())

important.remove(label)

cleaned_data = np.array([ result[x].tolist() for x in important ]).T


# print (result)
# print (cleaned_data)
#print(result.to_string())

groups = result.groupby(label)
number_of_classes = len(groups)  # Here we have 3 different classes
dictionary_of_sum = {}
numrber_of_features  = len(result.columns) -1 # We have feature 1 and feature 2 
sigma = 1
increament_current_row_in_matrix = 0



point_want_to_classify =[0.07000070000700007, 0.17017017017017017]

print(len(point_want_to_classify))

for k in range(1,number_of_classes+1):

	# 4.1 Initiate the sume to zero 
	dictionary_of_sum[k] = 0
	number_of_data_point_from_class_k = len(groups.get_group(k))

	# ** PATTERN LAYER OF PNN **
	# 5. Loop via the number of training example in class i 
	# 5.1 - Declare a temporary variable to hold the sum of gaussian distribution sum
	temp_summnation = 0.0

	# 6. Loop via number of points in the class - NUMBER OF POINTS IN THE CLASS!
	for i in range(1,number_of_data_point_from_class_k+1):

		# 6.1 - Implementation of getting Gaussians 

		# 6.1 - Implementation of getting Gaussians 
		tempx = (point_want_to_classify[0] - cleaned_data[increament_current_row_in_matrix][0]) * (point_want_to_classify[0] - cleaned_data[increament_current_row_in_matrix][0]) 
		tempy = (point_want_to_classify[1] - cleaned_data[increament_current_row_in_matrix][1]) * (point_want_to_classify[1] - cleaned_data[increament_current_row_in_matrix][1]) 
		temp_sum = -1 * (tempx + tempy)
		print("x",tempx, cleaned_data[increament_current_row_in_matrix][0],cleaned_data[increament_current_row_in_matrix][1])
		print("y",tempy)
		print(temp_sum)
		temp_sum = temp_sum/( 2 * np.power(sigma,2) )

		# 6.2 - Implementation of Sum of Gaussians
		temp_summnation = temp_summnation + temp_sum

		# 6.3 - Increamenting the row of the matrix to get the next data point
		increament_current_row_in_matrix  = increament_current_row_in_matrix + 1

	# 7. Finally - For K class - the Probability of current data point belonging to that class
	dictionary_of_sum[k]  = temp_summnation 


# 8. Get the classified class 
print (dictionary_of_sum)
classified_class = str( max(dictionary_of_sum, key=dictionary_of_sum.get) )
print (classified_class)

import numpy as np
import numpy.linalg 
import scipy
import cython
import matplotlib.pyplot as plt

epoch = np.arange(200) #create the array range = 0...199



read_two_cols = np.loadtxt('data2.txt', usecols=(0,1))
read_desired_ouput = np.loadtxt('data2.txt', usecols=2)
add_col = np.ones((200,1))
data= np.concatenate((add_col,read_two_cols),1)
weights_array = np.zeros(shape=(200,3))
MSE_Array = np.zeros(shape=(200,1))
p = 200


 

def LMS(desired_signal,learning_rate):
	# initialize Weight value [0,1,2]
	weights = np.array([0,1,2])
	print weights
	error = np.zeros(200)
	error_absolute = np.zeros(200)
	

	#LMS algorithm
	for n in range(200):
		error[n] = desired_signal[n] - weights.T.dot(data[n])
		weights = weights + learning_rate*data[n]*error[n]
		weights_array[n] = weights # add weight value to a new array
	

		
	# Mean-square- error vs epoch
	
	for j in range(200):
		diff =0.0
		for i in range(200):
			diff += (desired_signal[i]-weights_array[j].dot(data[i]))**2
	
		MSE = diff/(2*p)
	
		MSE_Array[j] = MSE
	

	
	#Plot the MSE vs epoch
	plt.plot(epoch,MSE_Array)	
	plt.show()

	return weights_array[np.argmin(MSE_Array)]
	


#learning rate = 0.01		
min_weight = LMS(read_desired_ouput, 0.01)
print 'learning rate =0.01 which has Weights value: '+str(min_weight)

#learning rate = 0.05
min_weight_2 = LMS(read_desired_ouput, 0.05)
print 'learning rate =0.05 which has Weights value: '+str(min_weight_2)
#learning rate = 0.1
min_weight_3= LMS(read_desired_ouput, 0.1)
print 'learning rate =0.1 which has Weights value: '+str(min_weight_3)




#Calculate the number of misclassification
stored_output = min_weight.reshape(min_weight.shape + (1,)).T.dot(data.T)
actual_output = np.array([])

for i in np.nditer(stored_output):
	if i >= 0:
		actual_output = numpy.append(actual_output,1)
	else:
		actual_output = numpy.append(actual_output,-1)

counter = 0 #counter for misclassification

#compare  actual_putput and desired_ouput array

for i, j in zip(np.nditer(actual_output),np.nditer(read_desired_ouput)):
	if i!=j:
		counter +=1

print 'number of misclassification: ' + str(counter)




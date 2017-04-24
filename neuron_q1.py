import numpy as np
import numpy.linalg 
import scipy
import cython
import matplotlib.pyplot as plt




M1 = np.array([[3],[4]]) # create the array for mean
M2 = np.array([[2],[3]])
C  = np.array([[1,0],[0,2]]) # create the array for C
C_inverse = numpy.linalg.inv(C) #inverse C
W_0 = 0.5*((M2.T).dot(C_inverse).dot(M2)-(M1.T).dot(C_inverse).dot(M1)) # Calculate b =W0
W_1_2 = C_inverse.dot(M1-M2) #calculate W=(w1,w2)
read_two_cols = np.loadtxt('data2.txt', usecols=(0,1))
print read_two_cols
read_desired_ouput = np.loadtxt('data2.txt', usecols=2)
#print read_desired_ouput

add_col = np.ones((200,1))
data= np.append(add_col,read_two_cols,1)
W = np.append(W_0,W_1_2,0)
print W

X = data.T
Y = W.T.dot(X) 
print Y

##iterate over output array to classify

actual_output = np.array([])


for i in np.nditer(Y):
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

#plot the scatter points with bayes classification and LMS classification
x = np.loadtxt('data2.txt', usecols=0)
y= np.loadtxt('data2.txt', usecols= 1)
x1 = numpy.linspace(-15,15,100)
y1 = x1*(-0.5) +4.25
y2 = x1*(0.00560961)/(-0.25895751) + (0.25895751/-0.25895751) # W=[-0.25895751 -0.00560961  0.14153423]
fig, ax= plt.subplots()
for output in np.unique(read_desired_ouput):
    mask = read_desired_ouput == output
    ax.plot(x[mask], y[mask], linestyle='none', marker='o', label=read_desired_ouput)
ax.plot(x1,y1)
ax.plot(x1,y2)
plt.show()



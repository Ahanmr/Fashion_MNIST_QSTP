import numpy as np
import sklearn          
import sklearn.datasets 
import matplotlib.pyplot as plt
import matplotlib.pyplot

# Initialize weights and biases
def init_weights_biases(input_dim, hdim1, hdim2, output_dim):
	# Initialize the parameters to random values. We need to learn these.
	np.random.seed(0)
	W1 = np.random.randn(input_dim, hdim1) / np.sqrt(input_dim)
	b1 = np.zeros((1, hdim1))
	W2 = np.random.randn(hdim1, hdim2) / np.sqrt(hdim1)
	b2 = np.zeros((1, hdim2))
	W3 = np.random.randn(hdim2, output_dim) / np.sqrt(output_dim)
	b3 = np.zeros((1, output_dim))
	return W1,b1,W2,b2,W3,b3

# Softmax Activation
def softmax(input):
	exp_scores = np.exp(input) 
	return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

# Cross Entropy Loss 
def log_likelihood(model, y):
	W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

	z1 = np.dot(X, W1) + b1
	# RELU
	# a1 = np.maximum(0, z1)
	# tanh
	a1 = np.tanh(z1)
	z2 = np.dot(a1, W2) + b2
	# RELU
	# a2 = np.maximum(0, z2)
	# tanh
	a2 = np.tanh(z2)
	z3 = np.dot(a2, W3) + b3
	probs = softmax(z3)

	# error = -1/len(probs)*np.sum(y*np.log(probs)+(1-y)*np.log(1-probs))
	error = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(error) / num_examples

	reg_loss = reg_strength/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	# return np.sum(np.nan_to_num(-y*np.log(probs)-(1-y)*n 
	return reg_loss + data_loss

def plot_decision_boundary(pred_func):
    # Set min and max values depending on data matrix and give it some padding-->(1)
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
    # Generate a grid of points with distance h between them, that will be the points we put the contour on-->(2)
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid, that gives the contour the color-->(3)
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)# makes contour plots of z, surface is xx and yy-->(4)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)#plots the normal scatter plot -->(5)
    #for contourplot reference see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.contourf

X, y = sklearn.datasets.make_moons(200, noise=0.20)

num_examples = len(X) # training set size
input_dim = 2         # input layer dimensionality
hdim1 = 3             # hidden layer1 
hdim2 = 3             # hidden layer2 
output_dim = 2        # output layer dimensionality

num_passes = 15000

# Gradient descent parameters 
lr_rate = 0.00001      # learning rate for gradient descent
reg_strength = 0.01 # regularization strength

# init weights and biases
W1, b1, W2, b2, W3, b3 = init_weights_biases(input_dim, hdim1, hdim2, output_dim)

# Gradient descent. For each batch
for i in range(0, num_passes):
	# Forward Propagation
	z1 = np.dot(X, W1) + b1
	# relu
	a1 = np.maximum(0, z1)
	# a1 = np.tanh(z1)
	z2 = np.dot(a1, W2) + b2
	# relu
	a2 = np.maximum(0, z2)
	# a2 = np.tanh(z2)
	z3 = np.dot(a2, W3) + b3
	probs = softmax(z3)

	# Backward Propagation
	delta3 = probs
	delta3[range(num_examples), y] -= 1
	delta3 /= num_examples
	dW3 = np.dot(a2.T, delta3)
	db3 = np.sum(delta3, axis=0, keepdims=True)

	# for RELU
	# delta2 = np.dot(delta3, W3.T)
	# delta2[a2 <=0 ] = 0
	delta2 = np.dot(delta3, W3.T) * (1-np.power(a2, 2))
	dW2 = np.dot(a1.T, delta2)
	db2 = np.sum(delta2, axis=0)

	# for RELU 
	# delta1 = np.dot(delta2, W2.T)
	# delta1[a1 <=0 ] = 0
	delta1 = np.dot(delta2, W2.T) * (1-np.power(a1, 2))
	dW1 = np.dot(X.T, delta1)
	db1 = np.sum(delta1, axis=0)

	# Add regularization 
	dW1 += reg_strength * W1
	dW2 += reg_strength * W2
	dW3 += reg_strength * W3

	# Gradient descent parameter update
	# update parameters with learning rate lr_rate
	W1 += lr_rate * W1
	b1 += lr_rate * b1
	W2 += lr_rate * W2
	b2 += lr_rate * b2
	W3 += lr_rate * W3
	b3 += lr_rate * b3

	model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}

	if i % 1000 == 0:
		# Calculate accuracy
		correct = [1 if i[0]>0 else 0 for i in probs]
		correct = [1 if a == b else 0 for (a, b) in zip(np.array(correct), y)]
		accuracy = np.sum(correct) / len(correct)

		print ("Loss after iteration {0}: {1} \t accuracy: {2}".format(i,log_likelihood(model, y), accuracy*100))

# plot_decision_boundary(lambda x: predict(model, x))
# plt.title("Decision Boundary for hidden layer size 3")
# plt.show()

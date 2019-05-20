import numpy as np 

''' Perceptron Neural Network (Single Layer)

Good explanation of logistic regression and its use cases:
https://sebastianraschka.com/faq/docs/logisticregr-neuralnet.html

This is an example (for personal learning) of how a one layer
logistic regression model can be created to predict binary outputs.

This was done following the youtube tutorial here:
https://www.youtube.com/watch?v=kft1AJ9WVDk&t=2s

Take a look at the video if you want to know the math and what 
equations are used for the synapses, neuron etc. and error functions.
It can probably do a lot better job that I can of explaining it, haha.

'''

def sigmoid(x):
	# Returns any value between 0 and 1 (Normalizing function)
	# based on x value.
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	# Derivative for error calculation. This is the derivative of the
	# above function.
	return x * (1 - x)



# These are training inputs, in this example we have three catgeories with
# binary values. For example, we could be trying to predict if it's sunny 
# based on the following data (0 for False, 1 for True):
# 
# | Is it cloudy? | Is it hot? | Am I wearing sunscreen? | RESULT: Is it sunny? |
#		no 				no                yes                     no
#		yes				yes				  yes 					  yes
#       yes				no  			  yes					  yes
# 
# etc. etc.
# The training outputs are the result (from the table above). Obviously the data
# in the table doesn't exactly corelate but it's just explaining what the 0,1 values
# are, as well as the number of features for our model.

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

# We start with random weights for each feature initially. The neuron calculates 
# the sum of all the features multiplied by their respective weights.
# ie. x1 (dot) w1 + x2w2 + x3w3 ... + xnwn
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights:')
print(synaptic_weights)

# Model training.
# We iterate many, many times to guarentee our outputs match our input.
# Training process:
# 	- Take the inputs from the training example and put them through the above
#		formula to get the neurons output
#	- Calculate the error, which is the difference between the output we got
#		and the actual output
#	- Depending on the severeness of the error, we adjust the weights accordingly.
#	- Repeat the process X times (a lot, to train the model effectively.)
#
# Forward propagation: We take the inputs, weights and put them through our summation
# Formula. 
# Back propogation: We calculate the error from our ouputs from the forward propagation
# and adjust the weights accordingly.

for iteration in range(20000):
	input_layer = training_inputs

	outputs = sigmoid(np.dot(input_layer, synaptic_weights))

	# Error weighted derivative.
	# We adjust the weights based on: error . input . derivative of sigmoid fn (output)
	# Error = output - actual output
	# Input = either 0 or 1
	# If the output of the error function was a large positive or negative value
	# (This is the slope of the sigmoid fn, the derivative), then the weight was heavy,
	# And this means that the neuron was very confident in it's output.
	# We don't want to mess with weights where the neuron was confident of it's output.
	# smaller numbers we adjust more, since the neuron is less confident about this output.

	error = training_outputs - outputs

	# Multiply how much we missed (error) by the slope of the sigmoid at the values in outputs.
	adjustments = error * sigmoid_derivative(outputs)

	# Update the weights accordingly.
	synaptic_weights += np.dot(input_layer.T, adjustments)


print('Synaptic weights after training:')
print(synaptic_weights)

print('Outputs after training:')
print(outputs)



from math import exp
from random import random
from csv import reader


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Getting Dataset ---------------------------------------------------------
# Creating dictionary
csv_reader = reader(open('dictionary.csv', 'r'))
dictionary = {}
for row in csv_reader:
    key = row[0]
    dictionary[key] = row[1]

# Getting training dataset from a CSV file
filename = "Sent.csv"
new_dataset = load_csv(filename)
dataset = []
vector = []
for sentence in new_dataset:
    for i in range(1, 6):
        vector.append(int(dictionary[sentence[i]]))
    dataset.append(vector)
    vector.append(int(sentence[0]))
    vector = []

# creating test data set
test_filename = 'testSamples.csv'
test_Samples = load_csv(test_filename)

test_dataset = []
test_vector = []
target = []
for sentence in test_Samples:
    test_vector = []
    target.append(int(sentence[5]))
    for i in range(0, 5):
        test_vector.append(int(dictionary[sentence[i]]))
    test_dataset.append(test_vector)


# Getting Dataset ---------------------------------------------------------


# Initialize a network ------------------------------------------------------------------------------
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Initialize a network ------------------------------------------------------------------------------


# Forward Propagate ---------------------------------------------------------------
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Forward Propagate ---------------------------------------------------------------


# Back Propagate Error ------------------------------------------------------------
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Back Propagate Error ------------------------------------------------------------


# Train Network --------------------------------------------------------------------------------
# Update network weights with error
def update_weights(Network, input_rows, l_rate):
    for i in range(len(Network)):
        inputs = input_rows[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in Network[i - 1]]
        for neuron in Network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Train Network --------------------------------------------------------------------------------

# Test Network ------------------------------------------------------------
# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    # return outputs.index(max(outputs))
    return outputs


# Test Network ------------------------------------------------------------

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)

train_network(network, dataset, 0.3, 5000, n_outputs)
print()
print("Testing...")
Expected = []
Got = []
count = 0
for row in test_dataset:
    prediction = predict(network, row)
    pos = prediction.index(max(prediction))
    if pos == 0:
        result = 0
    elif pos == 1:
        result = 1
    else:
        result = -1
    Expected.append(target[count])
    Got.append(result)
    # print('Expected=%d, Got=%d' % (row[-1], result))
    print('Expected=%d, Got=%d' % (target[count], result))
    count = count + 1

correct = 0
for i in range(len(Expected)):
    if Expected[i] == Got[i]:
        correct = correct + 1
total = len(Expected)

print()
accuracy = (correct/total)*100
accuracy = float("{0:.2f}".format(accuracy))
print()
print("Accuracy: " + str(accuracy) + " %")

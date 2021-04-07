import numpy 
# scipy.special for sigmoid function expit()
import scipy.special
# library for plotting arrays
# import matplotlib.pyplot
import time
start_time = time.time()
# nearal network class definition: init; train; query.
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 含input-hidden-output的3层神经网络，学习率learningrate
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.lr = learningrate 
        
        # weight inside the arrays are w_i_j, where link is from node i to node j in the next layer. 
        # easy but popular method to Initialize：
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        # self.who = (numpy.random.rand(self.onodes, self.hnodes)-0.5)
        
        #正态分布的方式实现初始权重：
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # activation function is the sigmoid function 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        
    
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emering from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # error = (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the inputs and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))
        
    
    def query(self, inputs_list):
        # convert inputs list to 2d array 
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #c alculate the signals emering from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    

# number of input, hidden and output nodes 
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
datafile = open("MNIST/mnist_train.csv", 'r')
datalist = datafile.readlines()
datafile.close()

# go through all the records in the training data set
# epochs is the number of times the training data set is used for training
epochs = 1
for e in range(epochs):
    print("training round {}...".format(e+1))
    for record in datalist:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01
        #creat the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) +0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    print(" time:", time.time()-start_time)

#load the mnist testing data CSV file into a list
test_data_file = open("MNIST/mnist_test.csv", 'r')
test_data_set = test_data_file.readlines()
test_data_file.close()
# test the nuaral network 
# scorecard for how well the network performs, initially empty 
scorecard = []
print("testing...")
for record in test_data_set:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is the first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct_label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01
    #creat the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) +0.01
    targets[int(all_values[0])] = 0.99
    # query the network 
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label 
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array  = numpy.asarray(scorecard)
print("performce = ", scorecard_array.sum()/scorecard_array.size)
print(" time:", time.time()-start_time)

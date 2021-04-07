import numpy as np  # data processing.
import scipy.special # for sigmoid function expit()
import time # for measuring time 
import csv

class NeuralNetwork:
    # 神经元的层数(int)(num=3),每层的个数(list)(layers[])，初始的学习因子(float)(lr)
    def __init__(self, layers=[1,1,1], lr=0.1, data_name="地火明夷"):
        # self.number = num
        self.layers = layers
        self.learningrate = lr
        self.data_path = "csvNetworks/%s.csv"%data_name
        self.number = len(self.layers)
        self.performce = 0
        # (layer-1)个正态分布初始权重
        try:
            self.Wijs = []
            with open(self.data_path, 'r',encoding='utf-8')as file:
                lines = file.readlines()
                i = 0
                for line in lines[2:]:
                    all_values = line.split(',')
                    # scale and shift the 
                    self.Wijs.append( np.asfarray(all_values).reshape(self.layers[i+1],self.layers[i]) )
                    i+=1 
            words = "Got Wijs-data from %s!"%(self.data_path)
        except:
            self.Wijs = []
            for i in range(self.number-1): 
                Wij = np.random.normal(0.0, pow(self.layers[i+1], -0.5), (self.layers[i+1], self.layers[i]) )
                self.Wijs.append(Wij) 
            words = "Generated new Wijs-data!" 
        
        # self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function = lambda x: scipy.special.expit(x)

        print('''The built NeuralNetwork:\n    layers: {};\n    learningrate: {};\n    data: {}\n    performance: {}\n'''.format(self.layers, self.learningrate, words, self.performce))

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array 
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # figure out the outputs for every layer
        output_list = [inputs] # number
        for i in range(self.number-1):
            output_list.append( self.activation_function(np.dot(self.Wijs[i], output_list[i])) )
        
        # the errors in layers, except for the first layer
        output_errors = targets - output_list[-1]
        error_list = [output_errors] # (number-1)
        for i in range(1,self.number-1):
            error_list.append( np.dot(self.Wijs[-i].T, error_list[i-1]) )
        error_list.reverse()

        # update the weights for the links between layers
        for i in range(self.number-1):
            self.Wijs[i] += self.learningrate * np.dot( (error_list[i]*output_list[i+1]*(1.0-output_list[i+1])), np.transpose(output_list[i]) )
        pass

    def query(self, inputs_list):
        # convert inputs list to 2d array 
        outputs = np.array(inputs_list, ndmin=2).T
        for wij in self.Wijs:
            outputs = self.activation_function(np.dot(wij, outputs))
        return outputs

    def save(self, name):
        file_name = "csvNetworks/%s.csv"%name
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f) 
            writer.writerow([name,'learningrate:',self.learningrate, 'layers:']) 
            writer.writerow(self.layers)
            for array in self.Wijs:
                writer.writerow(array.flatten().tolist())
    pass


def mnist_train(file, epochs):
    # load the mnist training data CSV file into a list
    datafile = open(file, 'r')
    datalist = datafile.readlines()
    datafile.close()
    # go through all the records in the training data set
    # epochs is the number of times the training data set is used for training
    print("train for %d epoch(s):"%epochs)
    for e in range(epochs):
        print("  training round {}...".format(e+1))
        for record in datalist:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:])/255.0 * 0.99)+0.01
            #creat the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(10) +0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        print("    time:", time.time()-start_time, "second")

def mnist_test(file):
    #load the mnist testing data CSV file into a list
    test_data_file = open(file, 'r')
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
        inputs = (np.asfarray(all_values[1:])/255.0 * 0.99)+0.01
        #creat the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(10) +0.01
        targets[int(all_values[0])] = 0.99
        # query the network 
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label 
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array  = np.asarray(scorecard)
    n.performce = scorecard_array.sum()/scorecard_array.size
    print("  performce = ", n.performce)
    print("    time:", time.time()-start_time, "second")

n = NeuralNetwork([784, 320, 72,10], 0.1, "乾坤新象")

train_file = "MNIST/mnist_train.csv"
test_file = "MNIST/mnist_test.csv"
epochs = 1
start_time = time.time()
mnist_train(train_file, epochs)
mnist_test(test_file)

n.save("乾坤新象")

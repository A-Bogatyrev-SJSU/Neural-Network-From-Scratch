# why can't java have a simple dependency manager, Maven sucks, 
# so does Gradle, half the packages are broken (same with python tbh)
# JCuda did not work so have to use python, also the fast java matrix 
# libraries have terrible syntax. It was either fortran or python, 
# my laziness prevailed
import numpy as np
import cupy
from scipy.special import expit

class NeuralNetwork:
    def __init__(self, num_in, layer_sizes, num_out, activation_function= "sigmoid", low = -1.0, high=1.0, learning_rate = .05):
        """constructor for Neural Network

        Args:
            num_in (int): input nodes
            layer_sizes (int []): list of nodes in each layer, expects length at least 1
            num_out (int): output nodes
            low (float, optional): starting values random number low value. Defaults to -1.0.
            high (float, optional): starting values random number high value. Defaults to 1.0.
            learning_rate (float, optional): learning rate of the neural network
        """
        self.num_in = num_in
        self.num_out = num_out
        self.layer_num = len(layer_sizes) # + 1 or + 2 because in/out layers?
        self.layer_list = list()
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.bias = list()
        
        for s in range(0,len(layer_sizes)):
            self.bias.append(np.zeros((layer_sizes[s],1)))
        self.bias.append(np.zeros((num_out,1)))    

        self.layer_list.append(np.random.uniform(low=low,high=high,size=(layer_sizes[0],num_in)))
        for i in range(0, len(layer_sizes)-1):
            self.layer_list.append(np.random.uniform(low=low,high=high,size=(layer_sizes[i+1],layer_sizes[i])))
        self.layer_list.append(np.random.uniform(low=low,high=high,size=(num_out,layer_sizes[-1])))

        # for el in self.layer_list:
        #     print(el)

        for el in self.layer_list:
            print(el.shape) 

    def train(self, input_array, target_array):
        # Reshape input and target arrays to column vectors
        inputs = np.array(input_array).reshape(-1, 1)
        targets = np.array(target_array).reshape(-1, 1)

        # get hidden raw
        hidden = np.dot(self.layer_list[0], inputs)
        hidden += self.bias[0]
        # Function
        if self.activation_function == "sigmoid":
            hidden = expit(hidden)
        else:
            #TODO other functions
            hidden = np.tanh(hidden)

        # get 2nd hidden raw
        outputs = np.dot(self.layer_list[1], hidden)
        outputs += self.bias[1]
        if self.activation_function == "sigmoid":
            outputs = expit(outputs)
        else:
            #TODO other functions
             outputs = np.tanh(outputs)

        #  error
        final_errors = targets - outputs

        # Calculate gradient
        if self.activation_function == "sigmoid":
            gradients = outputs * (1 - outputs)
        else:
            gradients =  1 - outputs ** 2
            # TODO
            pass
        gradients *= final_errors
        gradients *= self.learning_rate

        # deltas
        ho_deltas = np.dot(gradients, hidden.T)

        # Adjust weights
        self.layer_list[1] += ho_deltas
        # Adjust error
        self.bias[1] += gradients

        # hidden layer errors
        hidden_errors = np.dot(self.layer_list[1].T, final_errors)

        # Calculate hidden gradient
        if self.activation_function == "sigmoid":
            hidden_gradient = hidden * (1-hidden)
        else:
            hidden_gradient = 1 - hidden**2
            #TODO other derivatives
            pass
        hidden_gradient *= hidden_errors
        hidden_gradient *= self.learning_rate

        # deltas
        ih_deltas = np.dot(hidden_gradient, inputs.T)

        self.layer_list[0] += ih_deltas
        # fix bias
        self.bias[0] += hidden_gradient


    def process(self, input_array):
        # Reshape input array to a column vector
        inputs = np.array(input_array).reshape(-1, 1)

        # Generating the Hidden Outputs
        # print("input_array", input_array)
        # print("weights_ih:", self.layer_list[0])
        # print("inputs:", inputs)
        # print("self.bias_h", self.bias[0])
        hidden = np.dot(self.layer_list[0], inputs)
        # print("hidden", hidden)
        hidden += self.bias[0]
        # use activation func
        if self.activation_function == "sigmoid":
            hidden = expit(hidden)
        else:
            hidden = np.tanh(hidden)
            pass #TODO tanh

        # propogate
        output = np.dot(self.layer_list[1], hidden)
        output += self.bias[1]

        if self.activation_function == "sigmoid":
            output = expit(output)
        else:
            output = np.tanh(output)
            pass #TODO tanh 

        # flatten resultW
        return output.flatten()
    
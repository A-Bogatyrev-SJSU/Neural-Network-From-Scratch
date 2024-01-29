from NeuralNetwork import NeuralNetwork
import numpy as np
import random
#import cupy
#from scipy.stats import gmean
import matplotlib.pyplot as plt


def calculate_true_error(nn, samples=100, debug = False):
    errors = np.zeros(shape=(samples,1))
    for i in range(0,samples):
        w = np.random.uniform(low=-1.,high=1.)
        x = np.random.uniform(low=-1.,high=1.)
        y = np.random.uniform(low=-1.,high=1.)
        z = np.random.uniform(low=-1.,high=1.)
        
        #sin(x1-x2+x3-x4)
        sin = np.sin(w - x + y - z)


        nn_result = nn.process(np.array([w, x, y, z]))
        errors[i] = nn_result - sin
        if debug:
            print("sin(",w,"-",x,"+", y ,"-",z,") =", sin)
            print("actual sin(",w,"-",x,"+", y ,"-",z,") =", nn_result)
            print("error:", nn_result-sin, "actual:", nn_result,"expected:", sin)
       
    return np.average(np.abs(errors)), np.sqrt(np.mean(errors**2))

    pass

def calculate_test_error(nn, testdata, debug = False):
    errors = np.zeros(shape=(len(testdata),1))
    for i in range(0,len(testdata)):
        w = testdata[i][0][0]
        x = testdata[i][0][1]
        y = testdata[i][0][2]
        z = testdata[i][0][3]
        
        #sin(x1-x2+x3-x4)
        sin = testdata[i][1]

        
        nn_result = nn.process(np.array([w, x, y, z]))
        errors[i] = nn_result - sin
        if debug:
            print("sin(",w,"-",x,"+", y ,"-",z,") =", sin)
            print("actual sin(",w,"-",x,"+", y ,"-",z,") =", nn_result)
            print("error:", nn_result-sin, "actual:", nn_result,"expected:", sin)
       
    return np.average(np.abs(errors)), np.sqrt(np.mean(errors**2))

    pass

nn = NeuralNetwork(4, [14], 1,activation_function="tanh", low=-1, high=1, learning_rate=0.025)

data = []

for i in range(0,500):
    w = np.random.uniform(low=-1.,high=1.)
    x = np.random.uniform(low=-1.,high=1.)
    y = np.random.uniform(low=-1.,high=1.)
    z = np.random.uniform(low=-1.,high=1.)
    #sin(x1-x2+x3-x4)
    sin = np.sin(w-x+y-z)
    data.append((np.array([w,x,y,z]),np.array([sin])))


train = data [:401]
test = data [401:]
epochs = 1000

result_error_true = np.empty(epochs + 1)
result_error_test = np.empty(epochs + 1)
result_error_train = np.empty(epochs + 1)

for epoch in range(0,epochs+1):
    if epoch % 1 ==0:
        error_test = calculate_test_error(nn,test,debug=False)
        print(epoch, error_test)
        result_error_test[epoch]= error_test[0]
        result_error_train[epoch] = calculate_test_error(nn,train)[0]

    random.shuffle(data)
    for inp,out in data:
        nn.train(inp,out)
        pass

#print("Final true error avg(abs(err)) (500 samples):",calculate_true_error(nn,samples=500,debug=False)[0])
print("Final test error avg(abs(err)) (100 samples):",calculate_test_error(nn,test,debug=False)[0])
print("Final training data error avg(abs(err)) (400 samples):",calculate_test_error(nn,train,debug=False)[0])
x_values = np.arange(1, len(result_error_true) + 1)

fig, axs = plt.subplots(2)
axs[0].plot(x_values, result_error_train, marker='o', linestyle='-', color='b', label='Train Error')
axs[1].plot(x_values, result_error_test, marker='o', linestyle='--', color='r', label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs Epochs')
axs[0].set_yticks(np.arange(0,.7,.025))
axs[0].grid(color='g', linestyle='-', linewidth=1)
axs[0].legend()
axs[1].set_yticks(np.arange(0,.7,.025))
axs[1].grid(color='g', linestyle='-', linewidth=1)
axs[1].legend()
plt.show()
from NeuralNetwork import NeuralNetwork
import numpy as np
import random
# import cupy
#from scipy.stats import gmean
import matplotlib.pyplot as plt


def calculate_test_error(nn,data):
    size = len(data)
    sum = 0

    for input, output in data:
        result = nn.process(input)
        letter_nn = result.argmax()
        letter = output.argmax()

        if letter_nn == letter:
            sum += 1

    return 1.0 * sum / size
    
nn = NeuralNetwork(16, [150], 26, activation_function="sigmoid", low=-.1, high=.1, learning_rate=0.025)

data = []
file_path = "letter-recognition.txt"
try:
      with open(file_path, 'r') as file:
        for line in file:
            # Splitting the line by commas
            line_data = line.strip().split(',')
            letter = np.full(26, 0.)
            #print(letter)
            #print(line_data[0], ord(line_data[0])-ord('A'))
            letter[ord(line_data[0])-ord('A')] = 1.
            #print(letter)
            entry = (np.array(line_data[1:],dtype=float),letter)
            
            #print(entry)
            data.append(entry)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")

train = data[:len(data)*4//5]
test = data[len(data)*4//5:]

epochs = 1500

result_error_test = np.empty(epochs)
result_error_train = np.empty(epochs)

print(len(train), len(test))



for epoch in range (0,epochs):
    # will not shuffle this time because large
    for input, output in train:
        nn.train(input, output)

    if epoch % 1 ==0:
        result_error_train[epoch] = calculate_test_error(nn,train)
        result_error_test[epoch] = calculate_test_error(nn,test)
        print(epoch+1,"train_error:",result_error_train[epoch],"test_error:",result_error_test[epoch])
        
        # r = random.randint(0,len(train))
        # sample = train[1000]
        # result = nn.process(sample[0])
        # letter_nn = result.argmax()
        # letter = sample[1].argmax()
        # print(letter_nn, letter, result[letter], result[letter_nn])
        # print(result)



x_values = np.arange(1, epochs + 1)

fig, axs = plt.subplots(2)
axs[0].plot(x_values, result_error_train, marker='o', linestyle='-', color='b', label='Correctness Training Set')
axs[1].plot(x_values, result_error_test, marker='o', linestyle='--', color='r', label='Correctness Test Set')
plt.xlabel('Epoch')
plt.ylabel('% Correct')
plt.title('% Correct vs Epochs')
axs[0].set_yticks(np.arange(0,1.,.05))
axs[0].grid(color='g', linestyle='-', linewidth=1)
axs[0].legend()
axs[1].set_yticks(np.arange(0,1,.05))
axs[1].grid(color='g', linestyle='-', linewidth=1)
axs[1].legend()
plt.show()
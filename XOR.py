from NeuralNetwork import NeuralNetwork
#import cupy
import random
import numpy as np

nn = NeuralNetwork(2, [4], 1,activation_function="tanh", low=-2, high=2, learning_rate=0.1)

inputs = [np.array([1,1]),np.array([1,0]),np.array([0,0]),np.array([0,1])]
outputs =[np.array([0]),  np.array([1]),  np.array([0]),  np.array([1])]

print("here",nn.process(inputs[1]))

#nn.train_single_one_layer(inputs[1], outputs[1])

passed = False
for i in range(0,5000):
    r = random.randrange(0,4)
    nn.train(inputs[r], outputs[r])

    if i % 500 == 0:
        count = 0
        for k in range(0,4):
            out = nn.process(inputs[k])
            out_r = np.round(out)
            if out_r == outputs[k]:
                count+=1
                print("✓", "iteration:", i," input:",inputs[k]," output:", out, " output (rounded):", round(out.tolist()[0]), " expected:", outputs[k])
                pass
            else:
                print("✘", "iteration:", i," input:",inputs[k]," output:", out, " output (rounded):", round(out.tolist()[0]), " expected:", outputs[k])
                pass
            if count == 4 and not passed:
                passed = True
                print("Pass at:", i)
        print("-"*20)
        # for mat in nn.layer_list:
        #     print(mat)

for i in range(0,4):
    out = nn.process(inputs[i])
    out_r = np.round(out)
    print("✓" if out_r == outputs[i]  else "✘", "fin: input:",inputs[i]," output:", out, " output (rounded):", round(out.tolist()[0]), " expected:", outputs[i])
for mat in nn.layer_list:
    print(mat)
# for bias in nn.bias:
#     print(bias)

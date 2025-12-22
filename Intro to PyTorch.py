import torch 
# create tensors to store all of the numerical values 
# (inc. the raw data and the values for each weight and bias)
import torch.nn as nn
# make the weight and bias tensors part of the neural network
import torch.nn.functional as F
# give the activation functions
from torch.optim import SGD
# Stochastic Gradient Descent, to fit the neural network to the data
import matplotlib.pyplot as plt
import seaborn as sns

# With PyTorch, creating a new neural network means creating a new class
# BasicNN will inherit from a PyTorch class called Module
class BasicNN(nn.Module):

    # create an initialisation method for the new class
    def __init__(self):

        # call the initialisation method for the parent class, nn.Module
        super().__init__()

        # make a new variable and make it a neural network parameter
        # Making this weight a parameter gives us the option to optimise it.
        # initialise it with a tensor set to 1.7
        # Since it's a tensor, the neural network can take advantage of the accelerated arithmetic
        # and automatic differentiation that it provides.
        # As we don't need to optimise this weight,
        # set requires gradient to False
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        # pass input_to_top_relu to the ReLU activation function
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
# create as tensor with a sequence of 11 values between, and including, 0 and 1
input_doses = torch.linspace(start=0, end=1, steps=11)

input_doses

# make a neural network
model = BasicNN()

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values,
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()

class BasicNN_train(nn.Module):

    def __init__(self):

        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        # this parameter should be optimised
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)
    
    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        # pass input_to_top_relu to the ReLU activation function
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
model = BasicNN_train()

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values.detach(), 
             # create a new tensor that only has the output values (seaborn doesn't know what to do with the gradient)
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()

inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])

# optimise final_bias
# optimise every parameter which requires_grad=True
# lr: learning rate
optimiser = SGD(model.parameters(), lr=0.1)
# see how gradient descent improves the values for final_bias
print("Final bias, before optimisation: " + str(model.final_bias.data) + "\n")

# Gradient Descent
# Each time our optimisation code sees all of the training data is called an epoch
# run all 3 data points from the training data through the model up to 100 times.
for epoch in range(100):

    # store the loss, a measure of how well the model fits the data
    total_loss = 0

    # run each data point from the training data through the model and calculates the total loss
    for iteration in range(len(inputs)):

        input_i = inputs[iteration] # dose
        label_i = labels[iteration] # effectiveness

        output_i = model(input_i)

        loss = (output_i - label_i) ** 2 # MSELoss(), CrossEntropyLoss()...

        # calculate the derivative of the loss function wrt the parameter(s) we wanna optimise
        loss.backward()
        # It automatically accumulates the derivative each time we go through the nested loop!!!

        # keep track of how well the model fits all of the data
        total_loss += float(loss)

    if (total_loss < 0.0001):
        print("Num steps: " + str(epoch))
        break

    # take a small step towards a better value
    # It has access to the derivatives stored in model and can use them to step in the correct direction
    optimiser.step()
    # Zero out the derivatives stored in model
    optimiser.zero_grad()

    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")

print("Final bias, after optimisation: " + str(model.final_bias.data))

# Verify that the optimised model fits the training data by graphing it
output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values.detach(),
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
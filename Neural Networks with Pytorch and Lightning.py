import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader # good with larger datasets
from lightning.pytorch.tuner import Tuner

import matplotlib.pyplot as plt
import seaborn as sns

class BasicLightning(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = F.relu(input_to_final_relu)

        return output
    
input_doses = torch.linspace(start=0, end=1, steps=11)

input_doses

model = BasicLightning()

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values,
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()

class BasicLightningTrain(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # Use a lightning function to improve the learning rate
        self.learning_rate = 0.01 # just a placeholder value

    def forward(self, input):

        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

    def configure_optimizers(self):
        # set up the method we wanna use to optimise the neural network
        return SGD(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = (output_i - label_i) ** 2 # the for loop in pytorch reduced into this line of code

        return loss

model = BasicLightningTrain()

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values.detach(),
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()

# wrap the training data in a DataLoader
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])

# combine the inputs and the labels into a TensorDataset
dataset = TensorDataset(inputs, labels)
# use the tensor dataset ot create a DataLoader
dataloader = DataLoader(dataset)

# DataLoaders are useful when we have a lot of data:
# 1. They make it easy to access the data in batches
# 2. They make it easy to shuffle the data each epoch
# 3. They make it easy to use a relatively small fraction of data
# if we wanna do a quick and dirty training for debugging.

model = BasicLightningTrain()
# first use the trainer to find a good value for the learning rate
# and then use it optimise (train) the model 
trainer = L.Trainer(max_epochs=34)
# If we don't know the max number of epochs,
# Lightning lets us add additional epochs right where we left off

# Use the trainer to find an improved lr
tuner = Tuner(trainer)
lr_find_results = tuner.lr_find(model,
                                train_dataloaders=dataloader,
                                min_lr=0.001,
                                max_lr=1.0,
                                early_stop_threshold=None)
# By default, lr_find() will create 100 candidate lr btw the min and max values
# By setting early_stop_threshold to none, we'll test all of them
new_lr = lr_find_results.suggestion()
print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
model.learning_rate = new_lr

trainer.fit(model, train_dataloaders=dataloader)
print(model.final_bias.data)

output_values = model(input_doses)

sns.set(style="whitegrid")

sns.lineplot(x=input_doses,
             y=output_values.detach(),
             color='green',
             linewidth=2.5)

plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
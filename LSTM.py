import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# Adam is a lot like SGD, but not quite as stochastic.
# Adam finds the optimal values faster than SGD.

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyhand(L.LightningModule):
    
    def __init__(self):

        super().__init__()

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # use torch.normal() to initialise wlr1 to a number randomly generated
        # from a normal distribution with mean = 0 and sd = 1
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Do the LSTM math
    # long_memory: current long-term memory value
    def lstm_unit(self, input_value, long_memory, short_memory):

        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) +
                                              self.blr1)
        
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        
        updated_long_memory = ((long_memory * long_remember_percent) +
                               (potential_remember_percent * potential_memory))
        
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)
        
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return([updated_long_memory, updated_short_memory])

    # make a forward pass through unrolled LSTM
    def forward(self, input):

        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)
        
        return short_memory

    # configure Adam optimizer
    def configure_optimizers(self):
        return Adam(self.parameters())
        
    # calculate loss and log training process
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)
        # Lightning will create a new file in a directory called lightning_logs
        # and store whatever we wanna keep track of in it.

        # Keep track of the predictions of Company A and Company B
        if (label_i == 0): # made a prediction for Company B
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
    
model = LSTMbyhand()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
# Use the model to make and print a prediction for Company A
# by passing it a tensor containing the values from Days 1 through 4
# Call detach() to strip off the gradient

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# Horrible!!!

inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
# The labels are what we want the LSTM to predict for each company (on Day 5)
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=2000)
trainer.fit(model, train_dataloaders=dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
# worse than what we started with

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# a little than before

# self.log("train_loss", loss)
# if (label_i == 0):
#     self.log("out_0", output_i)
# else:
#     self.log("out_1", output_i)
# return loss
# We wrote the loss and the predictions to log filed in training_step()
# and that means we can use TensorBoard to draw graphs that tell us what happened
# during training and give us a sense of whether or not we should try to train more.

# It suggests we should do more training.

# Lightning saves checkpoints and they allow us to add additional epochs at any point during training.
path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path

trainer = L.Trainer(max_epochs=3000)
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)
# ckpt_path: checkpoint path

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path

trainer = L.Trainer(max_epochs=5000)
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)
# ckpt_path: checkpoint path

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# Every curve has flattened out, training done

class LightningLSTM(L.LightningModule):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=1)
        # input_size: no of features or variables
        # hidden_size: no of outputs
        # It's common to feed the output from the LSTM into the inputs of other neural network

    def forward(self, input):

        # transpose the input from one of the companies for Days 1 through 4
        # from being in a single row to being in a single column with view()
        input_trans = input.view(len(input), 1)
        # Specify the no of rows and one row per value/datapoint (len(input))
        # in order to create a single column with 4 values.
        # Then we specify the no of cols we want.
        # Since we have one feature and one col only, we set it to 1.

        lstm_out, temp = self.lstm(input_trans)
        # Save the output value in lstm_out.
        # lstm_out contains the short-term memory values from each LSTM unit we unrolled.
        # In this case, it has 4 values, as we needed to unroll the lstm 4 times for the 4 input values.

        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)
        
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
    
model = LightningLSTM()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
# Since we increased the lr, 
# we took bigger steps towards the optimal weights and biases each epoch,
# we set it to only do 300 epochs.
# By default, Lightning updates the log files every 50 steps.

trainer.fit(model, train_dataloaders=dataloader)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Company B: Observed = 1, Predicted =",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# The results have stabilised as each curve has flattened out.
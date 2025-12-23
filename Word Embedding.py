import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform # initialise weights in the neural network
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

inputs = torch.tensor([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])

labels = torch.tensor([[0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.],
                       [0., 1., 0., 0.]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

class WordEmbeddingFromScratch(L.LightningModule):

    def __init__(self):

        super().__init__()
        # required whenever we inherit from a class in Python

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        # input is a list that contains the One-Hot Encoding for one of the input tokens
        input = input[0]

        inputs_to_top_hidden = ((input[0] * self.input1_w1) +
                                (input[1] * self.input2_w1) +
                                (input[2] * self.input3_w1) +
                                (input[3] * self.input4_w1))
        
        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) +
                                   (input[1] * self.input2_w2) +
                                   (input[2] * self.input3_w2) +
                                   (input[3] * self.input4_w2))
        
        # Since the activation functions are identity functions...
        output1 = ((inputs_to_top_hidden * self.output1_w1) +
                   (inputs_to_bottom_hidden * self.output1_w2))
        output2 = ((inputs_to_top_hidden * self.output2_w1) +
                   (inputs_to_bottom_hidden * self.output2_w2))
        output3 = ((inputs_to_top_hidden * self.output3_w1) +
                   (inputs_to_bottom_hidden * self.output3_w2))
        output4 = ((inputs_to_top_hidden * self.output4_w1) +
                   (inputs_to_bottom_hidden * self.output4_w2))
        
        output_presoftmax = torch.stack([output1, output2, output3, output4])
        # If we just returned a list of the output ie [output1, ..., output4],
        # then the gradients would get stripped off and we would not be able to do backpropagation.
        # By using torch.stack(), we can return a list that preserves the gradients.
        return(output_presoftmax)

    def configure_optimizers(self):

        return Adam(self.parameters(), lr=0.1)
        # lr=0.1 cuz the example is simple and I wanna train it relatively quickly

    def training_step(self, batch, batch_idx):

        # split the batch of training data into the input and the labels (the ideal output values)
        input_i, label_i = batch
        # run the input thru the network up to the SoftMax() by passing it to the forward() method
        output_i = self.forward(input_i)
        # run the values forward() returns along with the ideal values thru the loss function
        # nn.CrossEntropyLOss() then runs the output values thru a SoftMax() function
        # and quantifies the difference between the SoftMax() output and the ideal values (0s and 1)
        # The difference is saved in loss.
        loss = self.loss(output_i, label_i[0])

        return loss
        # nn.CrossEntropyLoss() does the SoftMax for us

modelFromScratch = WordEmbeddingFromScratch()

print("Before optimisation, the parameters are...")
for name, param in modelFromScratch.named_parameters():
    print(name, param.data)

# put the Weight values into a dictionary
data = {
    # the Weight values for each input that goes to the activation function on top
    "w1": [modelFromScratch.input1_w1.item(),
           modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(),
           modelFromScratch.input4_w1.item()],
    # the Weight values for each input that goes to the activation function on the bottom       
    "w2": [modelFromScratch.input1_w2.item(),
           modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(),
           modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
# Use the item() method to get the Weights cuz it returns the tensor values as Python numbers
# Transform data into a Pandas Dataframe...
df = pd.DataFrame(data)
df
# Weights for Troll 2 and Gymkata are relatively different
# tho they both represent movie titles that are used in the same context
# To view in graph...
sns.scatterplot(data=df, x="w1", y="w2")
# add the tokens as labels to each point
plt.text(df.w1[0], df.w2[0], df.token[0],
         # pass in coordinates for the point and the value for the Token
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()
# the embedding values for Troll 2 and Gymkata are different
# so we need to train our embedding network

trainer = L.Trainer(max_epochs=100)
trainer.fit(modelFromScratch, train_dataloaders=dataloader)
data = {
    "w1": [modelFromScratch.input1_w1.item(),
           modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(),
           modelFromScratch.input4_w1.item()],
    "w2": [modelFromScratch.input1_w2.item(),
           modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(),
           modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
df
sns.scatterplot(data=df, x="w1", y="w2")
plt.text(df.w1[0], df.w2[0], df.token[0], # If the points overlap to each other, df.w1[0]-0.2
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()
# After training the Embedding network,
# the Embedding values for Troll 2 and Gymkata are very similar,
# which is great since they are used in similar contexts.

softmax = nn.Softmax(dim=0)
# dim=0 so that we can apply it to rows of output values
# dim=1, apply it to columns of values
print(torch.round(softmax(modelFromScratch(torch.tensor([[1., 0., 0., 0.]]))),
      # pass Troll 2 as a One-Hot Encoded tensor into modelFromScratch,
      # run the output value thru softmax()
                  decimals=2))
                  # round the output of the softmax to 2 decimal places
# Now we get the One-Hot Encoded tensor for is, which is correct!!!
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 1., 0., 0.]]))),
                  decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 1., 0.]]))),
                  decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 0., 1.]]))),
                  decimals=2))

class WordEmbeddingWithLinear(L.LightningModule):

    def __init__(self):

        # call the __init__ method from the parent class
        super().__init__()

        # create the weight between the inputs and the hidden layer
        # make 4 weights for each of 2 nodes in the hidden layer
        # we don't need any bias terms
        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        # give our class access to the Cross Entropy loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):

        # the linear object, input_to_hidden, does all of the * and + for us
        # now we are using linear to do the math
        # we no longer have to strip off the extra brackets from input like we did
        hidden = self.input_to_hidden(input)
        # the input to the activation functions is the same as the output, we ignore them
        output_values = self.hidden_to_output(hidden)

        # no need to calculate the softmax
        # nn.CrossEntropyLoss() does it for us alrdy
        return(output_values)
    
    def configure_optimizers(self):
        
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):

        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)

        return loss
    
modelLinear = WordEmbeddingWithLinear()

data = {
    # As we used nn.Linear() to create the Weights
    # we access them with .weight
    # .detach(): remove the gradient from the tensors
    # [0]: weights that go to the top activation functions
    # [1]: bottom
    # .numpy(): convert the tensor to a numpy array
    "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
    "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
df
sns.scatterplot(data=df, x="w1", y="w2")
plt.text(df.w1[0], df.w2[0], df.token[0], # If the points overlap to each other, df.w1[0]-0.2
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()
# not optimal yet

trainer = L.Trainer(max_epochs=100)
trainer.fit(modelLinear, train_dataloaders=dataloader)

data = {
    "w1": modelLinear.input_to_hidden.weight.detach()[0].numpy(),
    "w2": modelLinear.input_to_hidden.weight.detach()[1].numpy(),
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)

sns.scatterplot(data=df, x="w1", y="w2")
plt.text(df.w1[0]-0.2, df.w2[0]+0.1, df.token[0],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[1], df.w2[1], df.token[1],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[2], df.w2[2], df.token[2],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.text(df.w1[3], df.w2[3], df.token[3],
         horizontalalignment='left',
         size='medium',
         color='black',
         weight='semibold')

plt.show()

### load and use pre-trained Word Embedding values with nn.Embedding()

# access the Embedding values in modelLinear
modelLinear.input_to_hidden.weight # don't have to worry about the gradients
# the first row corresponds to the weights that go to the top activation function
# nn.Embedding() expects the weights to be in columns,
# just like in the dataframes

# pass the pre-trained weights in with .from_pretrained()
word_embeddings = nn.Embedding.from_pretrained(modelLinear.input_to_hidden.weight.T)
word_embeddings.weight
# print out the Embedding values for the first input
word_embeddings(torch.tensor(0))

# create a dictionary that maps the tokens to their indices
vocab = {'Troll2': 0,
         'is': 1,
         'great': 2,
         'Gymkata': 3}
word_embeddings(torch.tensor(vocab['Troll2']))

# we can now use Embedding() object, word_embeddings,
# and connect it to a larger neural network,
# like a Transformer!!!
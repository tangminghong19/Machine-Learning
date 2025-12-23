import torch # create tensors we'll use to store the raw data and to provide a few helper functions
import torch.nn as nn # for Module, Linear, Embedding classes, and other helper functions
import torch.nn.functional as F # access the softmax() function that we'll use when calculating attention

from torch.optim import Adam # fit the neural network to the data with backpropagation
from torch.utils.data import TensorDataset, DataLoader # tools to create large scale Transformer network with lots of training data

import lightning as L # automatic code optimisation and scaling in the cloud

# map the tokens to id numbers cuz the PyTorch word embedding function that we'll use, nn.Embedding(),
# only accepts numbers as input
token_to_id = {'who' : 0,
               'is' : 1,
               'MingHong' : 2,
               'handsome' : 3,
               '<EOS>' : 4,
               }

id_to_token = dict(map(reversed, token_to_id.items()))
# These dictionaries will make it easy to format the input to the transformer 
# and interpret the output from the Transformer

inputs = torch.tensor([[token_to_id["who"],
                        token_to_id["is"],
                        token_to_id["MingHong"],
                        token_to_id["<EOS>"],
                        token_to_id["handsome"]],
                        
                       [token_to_id["MingHong"],
                        token_to_id["is"],
                        token_to_id["who"],
                        token_to_id["<EOS>"],
                        token_to_id["handsome"]]])

labels = torch.tensor([[token_to_id["is"],
                        token_to_id["MingHong"],
                        token_to_id["<EOS>"],
                        token_to_id["handsome"],
                        token_to_id["<EOS>"]],
                        
                       [token_to_id["is"],
                        token_to_id["who"],
                        token_to_id["<EOS>"],
                        token_to_id["handsome"],
                        token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

### Position Encoding
# PE(pos, 2i) = sin(pos/10000^(2i/dmodel))
# PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel)
# dmodel: no of word embedding values we are using per token
# each token is specified with pos
# each embedding position is specified with i
# +1 means that the cos comes after the sin
# i: 0 0 1 1 ...

# precompute and add Position Encoding values to the tokens
class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):
        # d_model: dimension of the model, no of word embedding values per token
        # max_len: max no of tokens out transformer can process (input and output combined)
 
        super().__init__()

        # create a matrix of position encoding values
        pe = torch.zeros(max_len, d_model)

        # pos
        # use torch.arange() to create a sequence of numbers between 0 and max_len
        # .unsqueeze(1) turns the sequence of numbers into a column matrix
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        # create a row matrix that represents the index, i, times 2 (2i), for each word embedding
        # step=2 results in the same sequence numbers that we would get if i * 2
        # when d_model=2, embedding_index is just a single value 0 ie tensor([0.])
        # d_model = 6, tensor([0., 2., 4.])
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        # divisor
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        # assigns values from the sin function to the matrix pe
        # starting with the first col, col 0, and every col after that
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # use register_buffer() to ensure that pe gets moved to a GPU if we use one
        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):

        # add position encoding values to the word embedding values
        return word_embeddings + self.pe[:word_embeddings.size(0), :]
    
class Attention(nn.Module):

    def __init__(self, d_model=2):
    # we need to know the number of word embedding values per token
    # cuz that defines how large the weight matrices are that we use to create the Q, K, V

        super().__init__()

        # in_features: no of rows
        # out_features: cols
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        # A Linear() object doesn't just store the weights,
        # it will also do the math when the time comes
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # to give us flexibility to input training data in sequentially or in batches,
        # create some vars to keep track of which indices are for row and cols
        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
    # calculate the masked self-attention values for each token
    # for the sake of flexibility, allow the q, k, v to be calculated from different token encodings
    # so that we can do encoder-decoder attention
    # pass in a mask to do masked self-attention    

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        ### Attention(Q, K, V) = SoftMax((QK')/sqrt(dk) + M)*V

        # calculate the similarities btw the Q and the K
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5) # self.col_dim: no of values used in each key

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            # masking is used to prevent early tokens from cheating and looking at later tokens
            # -1e9, -infinity
            # create a final mask that is added to the scaled similarities

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        # determine the percentages of influence that each token should have on the others

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

class DecoderOnlyTransformer(L.LightningModule):
# Rather than having every class inherit from LightningModule
# allows us to take advantage of everything Lightning offers without the overhead of inheriting it multiple times

    def __init__(self, num_tokens=4, d_model=2, max_len=6):
    # num_tokens: no of tokens in the vocab
    # d_model: the no of values we want to represent each token
    # max_len: max length of input + output

        super().__init__()

        # Embedding() needs to know how many tokens are in the vocab
        # and the no of values we want to represent each token
        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model,
                                   max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        # fully connected layer
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()
        # use this as our model has multiple outputs
        # it will apply the softmax() function for us

    def forward(self, token_ids):
    # takes an array of token id numbers that will be used as inputs to the transformer

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        # create a matrix of 1s
        # torch.tril() leaves the values in the lower triangle
        mask = mask == 0
        # convert the 0s into Trues and 1s into Falses
        # eg. mask = tensor([[False,  True,  True],
        #                    [False, False,  True],
        #                    [False, False, False]]) for 3 token_ids

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)
        # the Q, K, V matrices will all be calculated from the same token encodings
        # thus pass in the same set of position encoded values 3 times for the Queries, Keys, and Values

        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
    def configure_optimizers(self):

        # pass all of the weights and biases in the model that we wanna train
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):

        # split the training data into inputs and labels
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        # compare the output from the Transformer to the known labels using the loss function
        loss = self.loss(output, labels[0])

        return loss
    
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)

model_input = torch.tensor([token_to_id["who"],
                            token_to_id["is"],
                            token_to_id["MingHong"],
                            token_to_id["<EOS>"]])
input_length = model_input.size(dim=0)
# Our model can only handle a total of 6 tokens
# so keeping track of how many tokens are in the input
# will tell us how many we can create as output

# generates predictions for each token in the input
predictions = model(model_input)
# only interested in what the model predicts after the <EOS> token
predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
# The outputs generated by the <EOS> token are an array of output values,
# one per possible output token.
# Thus use argmax to identify the output token with the largest value.
# The token with the largest output value will be the first token generated as a response to the input 
predicted_ids = predicted_id

# use a loop to keep generating output tokens
max_length = 6
for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break

    model_input = torch.cat((model_input, predicted_id))
    # ensure that each prediction is made with the full context

    # the model then predicts the next output token using the full context
    # which is the input plus the output tokens so far
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))
    
print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])

# train the model
trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

model_input = torch.tensor([token_to_id["who"],
                            token_to_id["is"],
                            token_to_id["MingHong"],
                            token_to_id["<EOS>"]])
input_length = model_input.size(dim=0)

predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
predicted_ids = predicted_id

max_length = 6
for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break

    model_input = torch.cat((model_input, predicted_id))

    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))
    
print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])

model_input = torch.tensor([token_to_id["MingHong"],
                            token_to_id["is"],
                            token_to_id["who"],
                            token_to_id["<EOS>"]])
input_length = model_input.size(dim=0)

predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
predicted_ids = predicted_id

max_length = 6
for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break

    model_input = torch.cat((model_input, predicted_id))

    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))
    
print("Predicted Tokens:\n")
for id in predicted_ids:
    print("\t", id_to_token[id.item()])

# Procedure:
# 1. Word Embedding
# 2. Position Encoding
# 3. Masked Self-Attention
# 4. Residual Connections
# 5. Fully Connected Layer
# 6. SoftMax()
import torch
import torch.nn as nn

import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

num_voices = 1
class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, latent_size, seq_size, num_layers):
        super(Encoder,self).__init__()
        self.RNN = nn.GRU(vocab_size, hidden_size, batch_first=True,
                        num_layers=num_layers, bidirectional = True)

        self.hidden_to_mu = nn.Linear(2*hidden_size, latent_size)
        self.hidden_to_sig = nn.Linear(2*hidden_size, latent_size)
        self.softplus = nn.Softplus()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.one_hot = torch.eye(vocab_size).to(device)


    def forward(self,x):
        batch_size = x.shape[0]

        _, h = self.RNN(self.one_hot[x])

        h = h.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size,-1)
        mu = self.hidden_to_mu(h)
        sig = self.softplus(self.hidden_to_sig(h))
        return mu, sig

    def flatten_parameters(self):
        self.RNN.flatten_parameters()

class Decoder(nn.Module):

    def __init__(self, latent_size, hidden_size, vocab_size, num_layers, seq_size):
        super(Decoder, self).__init__()
        #self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        #self.hidden_to_P1_hidden = nn.Linear(hidden_size, num_voices*num_layers*hidden_size)
        self.latent_to_hidden = nn.Linear(latent_size, num_voices*num_layers*hidden_size)
        self.hidden_to_vocab = nn.Linear(hidden_size, vocab_size)
        #self.P2_to_vocab = nn.Linear(hidden_size, vocab_size)

        self.RNN = nn.GRU(vocab_size + latent_size, hidden_size, batch_first = True, num_layers=num_layers)
        self.tanh = nn.Tanh()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.one_hot = torch.eye(vocab_size).to(device)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.seq_size = seq_size

    def forward(self, latent, temp, x, teacher_forcing, logits=False):
        batch_size = latent.shape[0]
        # latent = latent[indices]
        #hidden = self.tanh(self.latent_to_hidden(latent))

        hidden = self.tanh(self.latent_to_hidden(latent))
        hidden = hidden.view(self.num_layers, batch_size, -1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zeros = torch.zeros(batch_size,1).long().to(device)

        input = torch.cat((latent.unsqueeze(1),self.one_hot[zeros]),2)
        result = []

        if teacher_forcing:
            src_input = torch.cat((latent.unsqueeze(1).repeat(1,self.seq_size-1,1),self.one_hot[x[:,:-1]]),2)
            input = torch.cat((input, src_input),1)
            output, hidden = self.RNN(input, hidden)
            output = self.hidden_to_vocab(output)
            return output

        else:
            for i in range(self.seq_size):
                output, hidden = self.RNN(input, hidden)
                output = self.hidden_to_vocab(output)

                if temp:
                    res = Categorical(logits=   output/temp).sample()
                else:
                    res = torch.argmax(output, 2)
                if logits:
                    result.append(output)
                else:
                    result.append(res)
                if torch.rand(1) < teacher_forcing:
                    input = self.one_hot[x[:,i].unsqueeze(1)]
                else:
                    input =  self.one_hot[res]
                input = torch.cat((latent.unsqueeze(1), input),2)
        return torch.cat(result,1)

    def flatten_parameters(self):
        self.RNN.flatten_parameters()

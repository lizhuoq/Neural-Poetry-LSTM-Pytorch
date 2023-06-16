import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    '''
    input shape : seq_len , batch_size 
    output shape : seq_len , batch_size , vocab_size
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output)
        return output, hidden
    

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, target, invalid_idx):
        '''
        pred shape : batch_size , vocab_size , seq_len
        target shape : batch_size , seq_len
        '''    
        weights = torch.ones_like(target).fill_(invalid_idx)
        weights = (weights != target).float()
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, target)
        weighted_loss = (unweighted_loss*weights)
        return weighted_loss.mean()







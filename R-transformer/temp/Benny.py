import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import numpy as np

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module): # Input and Output dimensions: (batch_size, seq_length, d_model)
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x): # x: (batch_size, seq_len, d_model)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # Output: (batch_size, seq_len, d_model)

class SublayerConnection(nn.Module): # Input and Output dimensions: (batch_size, seq_length, d_model)
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # x: (batch_size, seq_len, d_model)
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x))) # Output: (batch_size, seq_len, d_model)


class PositionwiseFeedForward(nn.Module): # Input and Output dimensions: (batch_size, seq_length, d_model)
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = d_model * 4
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x: (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) # Output: (batch_size, seq_len, d_model)


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute 'Scaled Dot Product Attention'
        query, key, value : batch_size, n_head, seq_len, dim of space
    """
 
    d_k = query.size(-1)
    # scores: batch_size, n_head, seq_len, seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
   
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)


    p_attn = F.softmax(scores, dim = -1) # (batch_size, n_head, seq_len, seq_len)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn#Output:(batch_size,n_head,seq_len, d_k),(batch_size,n_head,seq_len,seq_len)


class SimpleCTRNN(nn.Module):
    """
    Simplified Continuous Time Recurrent Neural Network (CTRNN).
    """

    def __init__(self, input_size, hidden_size):
        super(SimpleCTRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.randn(1, hidden_size))
        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.input_weight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.tau = nn.Parameter(torch.randn(1, hidden_size).abs_())

    def forward(self, x, h0=None):
        batch_size, seq_len, input_dim = x.shape
        if h0 is None:
            h = torch.zeros((batch_size, self.hidden_size), device=x.device)
        else:
            h = h0
        outputs = []
        for i in range(seq_len):
            h = h + (1.0 / self.tau) * (-h + F.relu(h.mm(self.weight) + x[:, i, :].mm(self.input_weight) + self.bias))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        #auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask =  np.triu(np.ones(attn_shape), k=1).astype('uint8')
        # self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1)

    def forward(self, x):
        "Implements Figure 2"

        nbatches, seq_len, d_model = x.shape
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=self.mask[:,:, :seq_len, :seq_len], 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    # Input dimensions: (batch_size, seq_length, input_dim)
    # Output dimensions: (batch_size, seq_length, output_dim)
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'CTRNN':
            self.rnn = SimpleCTRNN(output_dim, output_dim)  # Replacing RNN with SimpleCTRNN
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = torch.LongTensor(idx)
        self.zeros = torch.zeros((self.ksize - 1, input_dim))

    # def forward(self, x):
    #     # Input dimensions: (batch_size, seq_length, input_dim)
    #     # Output dimensions: (batch_size, seq_length, ksize, input_dim)
    #
    #     # Input: x with dimensions (batch_size, seq_len, input_dim)
    #     nbatches, l, input_dim = x.shape
    #     x = self.get_K(x) # b x seq_len x ksize x d_model
    #     batch, l, ksize, d_model = x.shape
    #     h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
    #     # Output: h with dimensions (batch_size, seq_len, output_dim)
    #     return h.view(batch, l, d_model)
    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x) # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, ksize, d_model))  # Reshape input for CTRNN
        h_last = h[:, -1, :].view(batch, l, d_model)  # Select the last output and reshape
        return h_last

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])
        key = key.view(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x


class Block(nn.Module):
    """
    One Block
    """
    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.connections[0](x, self.pooling)
        x = self.connections[1](x, self.feed_forward)
        return x
    


class RTransformer(nn.Module):
    """
    The overall model:

    --rnn_type: The type of recurrent neural network to use for the local RNN layer.
    --d_model: The size of the input and output embeddings (dimensionality of the model).
    --n: The number of local RNN layers per block.
    --h: The number of attention heads in the multi-head attention pooling layer
    --ksize: The kernel size for the local recurrent neural network.
    --n_level: The number of layers (blocks) in the model.
    --dropout: The probability of dropout for regularization. This is the fraction of
    the input units to drop during training.

    Extra Info:

    ksize refers to the kernel size, which is the size of the local receptive field of the RNNs.
     In the implementation provided, a LocalRNN module is used to perform convolution-like operations
     with kernel size ksize. Specifically, it pads the input sequence and extracts local patches of size
     ksize to be used as inputs to the RNN. The output of each patch is then concatenated and returned as
     the output of the LocalRNN.

    n_level refers to the number of blocks in the model, while the number of local RNN layers per
    block is determined by the n parameter. Each block in the model is composed of a stack of local
    RNN layers followed by two sublayers: a multi-head attention pooling sublayer and a position-wise
    feedforward sublayer. By stacking multiple blocks, the model can learn more complex relationships
    between the inputs.

    h refers to the number of heads in the multi-head attention pooling sublayer. This sublayer performs
    self-attention on the input sequence by creating multiple attention weights for each position in the
    sequence. The heads allow the model to capture multiple relationships between the input positions,
    which can improve its performance on a range of tasks. A higher number of heads can provide
    greater expressiveness but also requires more computational resources.

    """
    def __init__(self, d_model, rnn_type, ksize, n_level, n, h, dropout):
        super(RTransformer, self).__init__()
        N = n
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, dropout)
        
        layers = []
        for i in range(n_level):
            layers.append(
                Block(d_model, d_model, rnn_type, ksize, N=N, h=h, dropout=dropout))
        self.forward_net = nn.Sequential(*layers) 

    def forward(self, x):
        x = self.forward_net(x)
        return x












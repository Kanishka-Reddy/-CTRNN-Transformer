import torch
from torch import Tensor
from torch.nn import RNNBase
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super(RNN, self).__init__(mode, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    hx = hx.unsqueeze(1)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            if self.mode == 'RNN_TANH':
                result = torch._C._VariableFunctions.rnn_tanh(input, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = torch._C._VariableFunctions.rnn_relu(input, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            if self.mode == 'RNN_TANH':
                result = torch._C._VariableFunctions.rnn_tanh(input, batch_sizes, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional)
            else:
                result = torch._C._VariableFunctions.rnn_relu(input, batch_sizes, hx, self._flat_weights, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional)

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)

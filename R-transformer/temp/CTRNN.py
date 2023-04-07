import math
import torch
import torch.nn as nn


class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activation='tanh', batch_first=False):
        super(CTRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        self.W_input = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hidden = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()

        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=input.dtype, device=input.device)

        output = []
        for t in range(seq_len):
            x_t = input[t]
            hx = self.activation(x_t @ self.W_input.t() + hx @ self.W_hidden.t() + self.bias)
            output.append(hx)

        output = torch.stack(output)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hx


# Example usage:
ctrnn = CTRNN(input_size=10, hidden_size=20, num_layers=1, activation='tanh', batch_first=False)
input = torch.randn(5, 3, 10)
output, hn = ctrnn(input)
print(output, hn)

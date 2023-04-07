import torch
from torch import nn


class CfCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = nn.Parameter(torch.randn(hidden_size, input_size))
        self.hidden_weights = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input, hidden_state, tau=1.0):
        pre_activation = (torch.matmul(input, self.input_weights.t()) +
                          torch.matmul(hidden_state, self.hidden_weights.t()) + self.bias)
        new_hidden_state = hidden_state + (-1.0 / tau) * (hidden_state - torch.tanh(pre_activation))
        return new_hidden_state, new_hidden_state


class SimpleCTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleCTRNN, self).__init__()

        self.rnn_cell = CfCCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state):
        hidden_state, _ = self.rnn_cell(input, hidden_state)
        output = self.fc(hidden_state)
        return output, hidden_state

# Example usage:
ctrnn = SimpleCTRNN(input_size=10, hidden_size=20, num_layers=1, activation='tanh', batch_first=False)
input = torch.randn(5, 3, 10)
output, hn = ctrnn(input)
print(output, hn)
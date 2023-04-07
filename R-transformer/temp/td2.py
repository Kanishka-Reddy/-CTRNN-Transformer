try:
    import torch
except:
    raise ImportWarning(
        "It seems like the PyTorch package is not installed\n"
        "Installation instructions: https://pytorch.org/get-started/locally/\n",
    )
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union
class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)
class CfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.
        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.CfC`.
        :param input_size:
        :param hidden_size:
        :param mode:
        :param backbone_activation:
        :param backbone_units:
        :param backbone_layers:
        :param backbone_dropout:
        :param sparsity_mask:
        """

        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                backbone_activation(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, new_hidden

from typing import Optional, Union


class WiredCfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        mode="default",
    ):
        super(WiredCfCCell, self).__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring

        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # Hack: nn.Module registers child params in set_attribute
            rnn_cell = CfCCell(
                in_features,
                len(hidden_units),
                mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx, timespans):
        h_state = torch.split(hx, self.layer_sizes, dim=1)

        new_h_state = []
        inputs = input
        for i in range(self.num_layers):
            h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
            inputs = h
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)
        return h, new_h_state

class CfC(nn.Module):
    def __init__(
        self,
        input_size: Union[int, ncps.wirings.Wiring],
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.
        Examples::
             >>> from ncps.torch import CfC
             >>>
             >>> rnn = CfC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)
        :param input_size: Number of input features
        :param units: Number of hidden units
        :param proj_size: If not None, the output of the RNN will be projected to a tensor with dimension proj_size (i.e., an implict linear output layer)
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        """

        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, ncps.wirings.Wiring):
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            # self.rnn_cell = WiredCfCCell(input_size, wiring_or_units)
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim
            self.rnn_cell = WiredCfCCell(
                input_size,
                self.wiring_or_units,
                mode,
            )
        else:
            self.wired_false = True
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units,
                backbone_layers,
                backbone_dropout,
            )
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        """
        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should "
                        f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                # batchless  mode
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            # batchless  mode
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx
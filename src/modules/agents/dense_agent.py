import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DenseAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DenseAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.dense_size)
        self.fc2 = nn.Linear(args.dense_size, args.dense_size)
        self.fc3 = nn.Linear(args.dense_size, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)

        x1 = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x1))
        q = self.fc3(x2)
        return q, hidden_state

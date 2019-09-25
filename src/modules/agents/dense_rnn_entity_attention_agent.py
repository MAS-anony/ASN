import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DenseRNNEntityAttentionAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DenseRNNEntityAttentionAgent, self).__init__()
        self.args = args

        assert self.args.enemy_feats_size == self.args.agent_feats_size, 'should expand to diff size'

        self.agent_enemy_num = args.enemies_num + args.agents_num - 1

        self.before_entity_size = args.move_feats_size
        self.after_entity_size = input_shape - args.move_feats_size - self.agent_enemy_num * self.args.enemy_feats_size

        self.entity_attention = nn.Linear(input_shape, self.agent_enemy_num)
        self.fc2 = nn.Linear(input_shape, args.dense_size)

        self.rnn3 = nn.GRUCell(args.dense_size, args.dense_size)
        self.fc4 = nn.Linear(args.dense_size, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2.weight.new(1, self.args.dense_size).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        # print(hidden_state.shape)

        self_attention_part_before = th.ones(inputs.size()[0], self.before_entity_size).cuda()# GPU version
        self_attention_part_after = th.ones(inputs.size()[0], self.after_entity_size).cuda()# GPU version

        attention_part = self.entity_attention(inputs)
        attention_part = attention_part.unsqueeze(-1).expand(
            [*attention_part.size(), self.args.enemy_feats_size]).reshape(attention_part.size()[0], -1)

        # print(self_attention_part_before.size(), attention_part.size(), self_attention_part_after.size())

        attention_part = th.cat([self_attention_part_before, attention_part, self_attention_part_after], dim=-1)


        x2 = F.relu(self.fc2(inputs * attention_part))

        h_in = hidden_state.reshape(-1, self.args.dense_size)
        h = self.rnn3(x2, h_in)

        q = self.fc4(h)
        # print(q.shape)
        return q, h

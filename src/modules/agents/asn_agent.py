import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AsnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AsnAgent, self).__init__()
        self.args = args

        # feature index
        self.move_feat_end = args.move_feats_size

        self.blood_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + args.agent_feats_size * (self.args.agents_num - 1)
        # self.blood_feat_start = 5 + 5 * 8 + 5 * 8
        self.blood_feat_end = self.blood_feat_start + 1

        self.other_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + args.agent_feats_size * (self.args.agents_num - 1) + 1
        # self.other_feat_start = 5 + 5 * 8 + 5 * 8 + 1

        self.enemies_feat_start = args.move_feats_size

        self.agents_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num


        # network struct
        self.env_info_fc1 = nn.Linear(input_shape, args.asn_hidden_size)
        self.env_info_fc2 = nn.Linear(args.asn_hidden_size, args.asn_hidden_size)

        # no-op + stop + up, down, left, right
        self.wo_action_fc = nn.Linear(args.asn_hidden_size, 6)

        self.enemies_info_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size)
        self.enemies_info_fc2 = nn.Linear(args.asn_hidden_size, args.asn_hidden_size)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.env_info_fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        enemies_feats = [inputs[:, self.enemies_feat_start + i * self.args.enemy_feats_size: self.enemies_feat_start + self.args.enemy_feats_size * (1 + i)]
                         for i in range(self.args.enemies_num)]


        env_hidden_1 = F.relu(self.env_info_fc1(inputs))
        env_hidden_2 = self.env_info_fc2(env_hidden_1)

        wo_action_fc_Q = self.wo_action_fc(env_hidden_2)

        enemies_hiddent_1 = [F.relu(self.enemies_info_fc1(enemy_info)) for enemy_info in enemies_feats]
        enemies_hiddent_2 = [self.enemies_info_fc2(enemy_info) for enemy_info in enemies_hiddent_1]

        attack_enemy_id_Q = [th.sum(env_hidden_2 * enemy_info, dim=-1, keepdim=True) for enemy_info in enemies_hiddent_2]

        q = th.cat([wo_action_fc_Q, *attack_enemy_id_Q], dim=-1)
        return q, hidden_state

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AsnDiffWoShareAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AsnDiffWoShareAgent, self).__init__()
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

        print(args.type1_num, args.type2_num)
        print(args.enemy_feats_size)

        # network struct
        self.env_info_fc1 = nn.Linear(input_shape, args.asn_hidden_size)
        self.env_info_fc2 = nn.Linear(args.asn_hidden_size, args.asn_hidden_size)

        # no-op + stop + up, down, left, right
        self.wo_action_fc = nn.Linear(args.asn_hidden_size, 6)

        # For 2s3z
        self.enemies_info_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size // self.args.enemies_num)
        self.enemies_info_fc2 = nn.Linear(args.asn_hidden_size // self.args.enemies_num, args.asn_hidden_size)

        self.enemies_info2_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size // self.args.enemies_num)
        self.enemies_info2_fc2 = nn.Linear(args.asn_hidden_size // self.args.enemies_num, args.asn_hidden_size)

        self.enemies_info3_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size // self.args.enemies_num)
        self.enemies_info3_fc2 = nn.Linear(args.asn_hidden_size // self.args.enemies_num, args.asn_hidden_size)

        self.enemies_info4_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size // self.args.enemies_num)
        self.enemies_info4_fc2 = nn.Linear(args.asn_hidden_size // self.args.enemies_num, args.asn_hidden_size)

        self.enemies_info5_fc1 = nn.Linear(args.enemy_feats_size, args.asn_hidden_size // self.args.enemies_num)
        self.enemies_info5_fc2 = nn.Linear(args.asn_hidden_size // self.args.enemies_num, args.asn_hidden_size)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.env_info_fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        # self_input = th.cat([inputs[:, :self.move_feat_end],
        #                      inputs[:, self.blood_feat_start: self.blood_feat_end],
        #                      inputs[:, self.other_feat_start:]],
        #                     dim=1)

        enemies_feats = [inputs[:, self.enemies_feat_start + i * self.args.enemy_feats_size: self.enemies_feat_start + self.args.enemy_feats_size * (1 + i)]
                         for i in range(self.args.enemies_num)]

        #
        # agents_feats = [inputs[:, self.agents_feat_start + i * 5: self.agents_feat_start + 5 * (1 + i)]
        #                 for i in range(self.args.agents_num - 1)]

        env_hidden_1 = F.relu(self.env_info_fc1(inputs))
        env_hidden_2 = self.env_info_fc2(env_hidden_1)

        wo_action_fc_Q = self.wo_action_fc(env_hidden_2)

        enemies_hiddent_1 = []
        enemies_hiddent_2 = []

        # hand code for 2s3z
        enemies_hiddent_1.append(F.relu(self.enemies_info_fc1(enemies_feats[0])))
        enemies_hiddent_2.append(self.enemies_info_fc2(enemies_hiddent_1[0]))

        enemies_hiddent_1.append(F.relu(self.enemies_info2_fc1(enemies_feats[1])))
        enemies_hiddent_2.append(self.enemies_info2_fc2(enemies_hiddent_1[1]))

        enemies_hiddent_1.append(F.relu(self.enemies_info3_fc1(enemies_feats[2])))
        enemies_hiddent_2.append(self.enemies_info3_fc2(enemies_hiddent_1[2]))

        enemies_hiddent_1.append(F.relu(self.enemies_info4_fc1(enemies_feats[3])))
        enemies_hiddent_2.append(self.enemies_info4_fc2(enemies_hiddent_1[3]))

        enemies_hiddent_1.append(F.relu(self.enemies_info5_fc1(enemies_feats[4])))
        enemies_hiddent_2.append(self.enemies_info5_fc2(enemies_hiddent_1[4]))


        attack_enemy_id_Q = [th.sum(env_hidden_2 * enemy_info, dim=-1, keepdim=True) for enemy_info in enemies_hiddent_2]

        q = th.cat([wo_action_fc_Q, *attack_enemy_id_Q], dim=-1)
        return q, hidden_state

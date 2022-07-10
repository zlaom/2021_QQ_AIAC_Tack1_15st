import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.model_base import ModelBase

class NeXtVLAD(ModelBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.feature_size = cfg["FEATURE_SIZE"]
        self.output_size = cfg["OUTPUT_SIZE"]
        self.expansion_size = cfg["EXPANSION_SIZE"]
        self.cluster_size = cfg["CLUSTER_SIZE"]
        self.groups = cfg["NUM_GROUPS"]
        self.drop_rate = cfg["DROPOUT_PROB"]

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = nn.Dropout(self.drop_rate)
        self.expansion_linear = nn.Linear(self.feature_size,
                                          self.expansion_size * self.feature_size)
        self.group_attention = nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = nn.Linear(self.expansion_size * self.feature_size,
                                        self.groups * self.cluster_size,
                                        bias=False)
        self.cluster_weight = nn.Parameter(
            nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)
        # self.apply(weights_init_kaiming)

    def forward(self, inputs):
        # todo mask
        inputs = inputs.cuda()
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad
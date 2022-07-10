import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding

class ModelIntegration(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

        self.c = nn.Parameter(torch.ones(0.7, 256))
    
    def forward(self, frame_feature, mask, asr_title):
        _, embedding1 = self.model1(frame_feature, mask, asr_title)
        _, embedding2 = self.model2(frame_feature, mask, asr_title)

        weight = torch.sigmoid(self.c)
        c = self.c
        embedding = embedding1 * weight + embedding2 * (1 - weight)
        b, m = embedding.shape
        embedding_1, embedding_2 = embedding[:b // 2], embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, embedding


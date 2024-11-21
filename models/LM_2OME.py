import torch.nn as nn
import torch.nn.functional as F
from utils.config import *

params = config()

class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(AttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights


class LM_2OME(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super(LM_2OME, self).__init__(*args, **kwargs)

        self.bidirectional = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.attn_blk = AttentionBlock(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1)

        self.lin = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(10496, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.dense2 = nn.Sequential(
            nn.Linear(128, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.main_output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(params.device)

    def forward(self, input):
        w2v, embedding = input
        w2v = w2v.to(params.device)
        embedding = embedding.to(params.device)
        w2v, (h, c) = self.bidirectional(w2v)

        w2v = w2v.permute(0, 2, 1)
        w2v = F.interpolate(w2v, size=41, mode='linear', align_corners=False)
        w2v = w2v.permute(0, 2, 1)

        embedding = self.lin(embedding)

        embedding = embedding.permute(0, 2, 1)
        embedding = F.interpolate(embedding, size=41, mode='linear', align_corners=False)
        embedding = embedding.permute(0, 2, 1)

        concat = torch.cat((w2v, embedding), dim=2)

        outputs, att_weights = self.attn_blk(concat)
        # print(att_weights.shape)

        dense1_out = self.dense1(self.flatten(outputs))
        # dense1_out = dense1_out.unsqueeze(1)
        # self_att = self.self_att_seq(dense1_out)
        dense2_out = self.dense2(dense1_out)
        logits = self.main_output(dense2_out).squeeze(1)
        preds = self.sigmoid(logits)
        return preds, dense1_out, att_weights

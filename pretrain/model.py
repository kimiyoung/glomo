from torch import nn
from torch.nn.utils import rnn
from nn_utils import EncoderRNN, SA, ConvSA, PointerRNN, flip
from torch.autograd import Variable
import torch
import copy
import numpy as np

class Model(nn.Module):
    def __init__(self, d, ntokens, nlayers, nlayers_key, nlayers_query, use_value, dec_layers, dropout, dec_len, sample_len, kernel_size, use_gru, cnn_sa, yann, sparse_mode, topk_rate):
        super().__init__()
        self.enc_emb = nn.Embedding(ntokens, d, scale_grad_by_freq=True)
        self.enc_emb.weight.data.uniform_(-0.1, 0.1)
        self.cnn_sa = cnn_sa
        if self.cnn_sa:
            self.st_predictor_fw = ConvSA(d, d//2, kernel_size, nlayers, nlayers_key, nlayers_query, use_value, dropout, 8, use_gru, yann, sparse_mode, topk_rate)
            self.st_predictor_bw = ConvSA(d, d//2, kernel_size, nlayers, nlayers_key, nlayers_query, use_value, dropout, 8, use_gru, yann, sparse_mode, topk_rate)
        else:
            raise ValueError
            self.st_predictor_fw = SA(d, d//2, nlayers, dropout, 8, use_gru, yann)
            self.st_predictor_bw = SA(d, d//2, nlayers, dropout, 8, use_gru, yann)
        self.dec_net_fw = PointerRNN(d, dec_layers, dropout, ntokens)
        self.dec_net_bw = PointerRNN(d, dec_layers, dropout, ntokens)
        self.dec_len = dec_len
        self.sample_len = sample_len
        self.dec_net_fw.out.weight = self.enc_emb.weight
        self.dec_net_bw.out.weight = self.enc_emb.weight
        self.nlayers = nlayers

    def set_gru(self, use_gru):
        self.st_predictor_fw.use_gru = use_gru
        self.st_predictor_bw.use_gru = use_gru

    def forward(self, input, mask=None, return_graphs=False, return_outputs=False):
        bsz, slen = input.size(0), input.size(1)

        embeddings = self.enc_emb(input)
        loss = 0
        for dir in range(2):
            if dir == 1:
                embeddings_t = flip(embeddings, 1)
                mask_t = flip(mask, 1) if mask is not None else None
                input_t = flip(input, 1)
            else:
                embeddings_t, mask_t, input_t = embeddings, mask, input

            if dir == 0:
                graphs_fw, outputs_fw = self.st_predictor_fw(embeddings_t, mask_t)
                outputs = outputs_fw
            else:
                graphs_bw, outputs_bw = self.st_predictor_bw(embeddings_t, mask_t)
                outputs = outputs_bw
            if return_graphs or return_outputs:
                continue

            sampled_indices = np.random.choice(slen-self.dec_len, self.sample_len)
            sampled_indices = Variable(torch.from_numpy(sampled_indices).long().cuda())
            states = outputs[-1].index_select(1, sampled_indices)
            dec_net = self.dec_net_fw if dir == 0 else self.dec_net_bw
            loss += dec_net(states, sampled_indices, input_t, self.dec_len, embeddings_t) * 0.5

        if return_graphs:
            return torch.cat([graphs_fw, flip(flip(graphs_bw, -1), -2)], dim=0).permute(1, 0, 2, 3, 4).contiguous() # [B, LY, K, S, S]
        if return_outputs:
            return torch.cat([embeddings[None], outputs_fw, flip(outputs_bw, -2)], dim=0)

        return loss

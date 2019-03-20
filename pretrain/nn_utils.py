import sys
from torch import nn
from torch.nn import Parameter
import torch
import math
from  torch.nn import functional as F
from torch.autograd import Function, Variable
from torch.nn.utils import rnn

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class ConvSA(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, nlayers, nlayers_key, nlayers_query, use_value, dropout, K, use_gru, yann, sparse_mode, topk_rate):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.key_cnn = nn.ModuleList([
                nn.Sequential(nn.Conv1d(input_size, input_size, kernel_size, padding=kernel_size-1), nn.BatchNorm1d(input_size), nn.ReLU())
            for _ in range(nlayers_key)])
        self.query_cnn = nn.ModuleList([
                nn.Sequential(nn.Conv1d(input_size, input_size, kernel_size, padding=kernel_size-1), nn.BatchNorm1d(input_size), nn.ReLU())
            for _ in range(nlayers_query)])

        # self.key_dot_trans = nn.ModuleList([
        #         nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        #     for _ in range(nlayers)])
        # self.query_dot_trans = nn.ModuleList([
        #         nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        #     for _ in range(nlayers)])

        self.key_dot_trans = nn.ModuleList([
                nn.Linear(input_size, hidden_size*K)
            for _ in range(nlayers)])
        self.query_dot_trans = nn.ModuleList([
                nn.Linear(input_size, hidden_size*K)
            for _ in range(nlayers)])

        if not use_gru:
            if use_value == 1:
                self.value_trans = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(nlayers)])
        else:
            self.gru_cell = nn.GRUCell(input_size*2, input_size)
        self.use_gru = use_gru
        self.nlayers = nlayers
        self.nlayers_key = nlayers_key
        self.nlayers_query = nlayers_query
        self.use_value = use_value
        self.K = K
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.yann = yann
        self.sparse_mode = sparse_mode

        if sparse_mode == 1: # ReLU
            pass
        elif sparse_mode == 2: # shifted ReLU for each layer
            self.sparse_bias = nn.Parameter(torch.Tensor(self.nlayers).fill_(1e-4))
        elif sparse_mode == 3: # top K
            self.topk_rate = topk_rate
        elif sparse_mode == 4: # linear to compute shift
            self.sparse_trans = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(nlayers)])

        self.L, self.cache_mask, self.cache_diag = 0, None, None

    def get_cur_mask(self, input_len):
        #if input_len <= self.L:
        #    return Variable(self.cache_mask[:input_len, :input_len])
        self.cache_mask = torch.ones(input_len, input_len).cuda().tril(-1)
        self.L = input_len
        return Variable(self.cache_mask)

    def forward(self, input, mask=None):
        bsz, input_len = input.size(0), input.size(1)
        K = self.K
        output = input
        graphs, outputs = [], []
        cur_mask = self.get_cur_mask(input_len)
        # key conv
        key_output = self.dropout(output)
        for i in range(self.nlayers_key):
            if mask is not None:
                masked_output = key_output * mask[:,:,None]
            else:
                masked_output = key_output
            masked_output = masked_output.permute(0, 2, 1).contiguous()
            key_output = self.key_cnn[i](masked_output)[:,:,:-(self.kernel_size-1)]
            key_output = key_output.permute(0, 2, 1).contiguous()
        # query conv
        query_output = self.dropout(output)
        for i in range(self.nlayers_query):
            if mask is not None:
                masked_output = query_output * mask[:,:,None]
            else:
                masked_output = query_output
            masked_output = masked_output.permute(0, 2, 1).contiguous()
            query_output = self.query_cnn[i](masked_output)[:,:,:-(self.kernel_size-1)]
            query_output = query_output.permute(0, 2, 1).contiguous()

        if self.sparse_mode == 3:
            topk_trunc = min(max(int(math.sqrt(input_len) * self.topk_rate), 1), input_len)
        # run SA
        for i in range(self.nlayers):
            query_output_i = self.query_dot_trans[i](query_output).view(bsz, input_len, K, -1).permute(0, 2, 1, 3).contiguous() # [B, K, S, H]
            key_output_i = self.key_dot_trans[i](key_output).view(bsz, input_len, K, -1).permute(0, 2, 3, 1).contiguous() # [B, K, H, S]
            att = torch.matmul(query_output_i, key_output_i)
            if not self.yann:
                raise NotImplementedError
            else:
                if self.sparse_mode == 0:
                    att = att * cur_mask
                    if mask is not None:
                        att = att * mask[:,None,None]
                    att.pow_(2).div_((att.sum(dim=-1, keepdim=True) + 1e-16))
                elif self.sparse_mode == 1:
                    att = att.clamp(min=0.)
                    att = att * cur_mask
                    if mask is not None:
                        att = att * mask[:,None,None]
                    att.pow_(2).div_((att.sum(dim=-1, keepdim=True) + 1e-16))
                elif self.sparse_mode == 2:
                    att = (att - self.sparse_bias[i]).clamp(min=0.)
                    att = att * cur_mask
                    if mask is not None:
                        att = att * mask[:,None,None]
                    att.pow_(2).div_((att.sum(dim=-1, keepdim=True) + 1e-16))
                elif self.sparse_mode == 3:
                    att = att * cur_mask
                    if mask is not None:
                        att = att * mask[:,None,None]
                    att.pow_(2)
                    topk_indices = att.topk(k=topk_trunc, dim=-1, sorted=False)[1]
                    topk_mask = Variable(att.data.new(bsz, K, input_len, input_len).zero_().scatter_(-1, topk_indices.data, 1.0))
                    att = att * topk_mask
                    att.div_((att.sum(dim=-1, keepdim=True) + 1e-16))
                elif self.sparse_mode == 4:
                    sparse_th = self.sparse_trans[i](query_output_i)
                    att = (att - sparse_th).clamp(min=0.)
                    att = att * cur_mask
                    if mask is not None:
                        att = att * mask[:,None,None]
                    att.pow_(2).div_((att.sum(dim=-1, keepdim=True) + 1e-16))
                else:
                    raise NotImplementedError

            if not self.use_gru:
                if self.use_value == 1:
                    value_output = self.value_trans[i](output).view(bsz, input_len, K, -1).permute(0, 2, 1, 3) # [B, K, S, H]
                else:
                    value_output = output.view(bsz, input_len, K, -1).permute(0, 2, 1, 3) # [B, K, S, H]
            else:
                value_output = output.view(bsz, input_len, K, -1).permute(0, 2, 1, 3) # [B, K, S, H]
            cur_output = torch.matmul(att, value_output).permute(0, 2, 1, 3).contiguous().view(bsz, input_len, -1) # [B, S, H]

            if not self.use_gru:
                output = output + cur_output
            else:
                cur_output = torch.cat([cur_output * output, cur_output], dim=2)
                output = self.gru_cell(cur_output.view(bsz*input_len, -1), output.view(bsz*input_len, -1)).view(bsz, input_len, -1)

            graphs.append(att)
            outputs.append(output)

        return torch.stack(graphs, dim=0), torch.stack(outputs, dim=0)

class SA(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout, K, use_gru, yann):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.query_dot_trans = nn.ModuleList(
                [nn.Sequential(
                        nn.Linear(input_size, hidden_size*K),
                        nn.ReLU()
                    ) for _ in range(nlayers)]
            )
        self.key_dot_trans = nn.ModuleList(
                [nn.Sequential(
                        nn.Linear(input_size, hidden_size*K),
                        nn.ReLU()
                    ) for _ in range(nlayers)]
            )
        if not use_gru:
            self.value_trans = nn.ModuleList([GateLayer(input_size, input_size) for _ in range(nlayers)])
        else:
            self.gru_cell = nn.GRUCell(input_size*2, input_size)
        self.use_gru = use_gru
        self.nlayers = nlayers
        self.K = K
        self.hidden_size = hidden_size
        self.L, self.cache_mask = 0, None
        self.yann = yann

    def get_cur_mask(self, input_len):
        if input_len <= self.L:
            return Variable(self.cache_mask[:input_len, :input_len])
        self.cache_mask = torch.ones(input_len, input_len).cuda().tril(-1)
        self.L = input_len
        return Variable(self.cache_mask)

    def forward(self, input, mask=None):
        bsz, input_len = input.size(0), input.size(1)
        K = self.K
        output = input
        graphs, outputs = [], []
        cur_mask = self.get_cur_mask(input_len)
        for i in range(self.nlayers):
            output = self.dropout(output)
            query_output = self.query_dot_trans[i](output).view(bsz, input_len, K, -1).permute(0, 2, 1, 3).contiguous() # [B, K, S, H]
            key_output = self.key_dot_trans[i](output).view(bsz, input_len, K, -1).permute(0, 2, 3, 1).contiguous() # [B, K, H, S]
            att = torch.matmul(query_output, key_output) / (self.hidden_size ** 0.5)
            if not self.yann:
                att = att - 1e30 * (1 - cur_mask)
                if mask is not None:
                    att = att - 1e30 * (1 - mask[:, None, None])
                att = F.softmax(att, dim=-1)
            else:
                att = att * cur_mask
                if mask is not None:
                    att = att * mask[:,None,None]
                att.pow_(2).div_((att.sum(dim=-1, keepdim=True) + 1e-16))

            # TODO  
            raise NotImplementedError
            value_output = output.view(bsz, input_len, K, -1).permute(0, 2, 1, 3) # [B, K, S, H]
            cur_output = torch.matmul(att, value_output).permute(0, 2, 1, 3).contiguous().view(bsz, input_len, -1) # [B, S, H]
            
            if not self.use_gru:
                output = output + cur_output
            else:
                cur_output = torch.cat([cur_output * output, cur_output], dim=2)
                output = self.gru_cell(cur_output.view(bsz*input_len, -1), output.view(bsz*input_len, -1)).view(bsz, input_len, -1)

            graphs.append(att)
            outputs.append(output)
        return torch.stack(graphs, dim=0), torch.stack(outputs, dim=0)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class PointerRNN(nn.Module):
    def __init__(self, d, dec_layers, dropout, ntokens):
        super().__init__()
        self.rnn_fw = nn.GRU(d, d, dec_layers, batch_first=True, dropout=dropout)
        self.dropout = LockedDropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.dec_layers = dec_layers
        self.out = nn.Linear(d, ntokens, bias=False)
        self.ntokens = ntokens

    def forward(self, states, indices, word_indices, dec_len, embeddings):
        # states: [B, sample, H], indices: [sample], word_indices: [B, S], embeddings: [B, S, H]
        bsz, sample_len, d = states.size(0), indices.size(0), states.size(2)

        states = self.dropout(states)
        states = states.view(-1, d)[None].expand(self.dec_layers, -1, -1).contiguous() # [dec_layers, B*sample, H]

        inputs = []
        for i in range(dec_len):
            if i == 0:
                inputs.append(Variable(torch.Tensor(bsz, sample_len, d).cuda().zero_()))
            else:
                inputs.append(embeddings.index_select(1, indices + i))
        input = torch.stack(inputs, dim=2) # [B, sample, D, H]

        output = input.view(-1, dec_len, d) # [B*sample, D, H]
        output, _ = self.rnn_fw(output, states)
        logits = self.out(output).view(-1, self.ntokens) # [B*sample*D, V]

        targets = []
        for i in range(dec_len):
            targets.append(word_indices.index_select(1, indices + i + 1))
        target = torch.stack(targets, dim=2).contiguous().view(-1)

        loss = self.loss(logits, target)

        return loss


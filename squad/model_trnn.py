
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn

class ModelTRNN(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        self.use_elmo = config.use_elmo
        self.use_glove = config.use_glove
        self.word_dim = int(self.use_elmo) * config.elmo_dim + int(self.use_glove) * config.glove_dim
        assert self.word_dim > 0

        if self.use_glove:
            self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
            if not config.train_emb:
                self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
                self.word_emb.weight.requires_grad = False
        if self.use_elmo:
            self.elmo_scale = nn.Parameter(torch.Tensor(1).fill_(1.0))
            self.elmo_mixt_logits = nn.Parameter(torch.Tensor(3).zero_())
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden

        self.dropout = LockedDropout(1-config.keep_prob0)

        self.uniform_graph = config.uniform_graph
        self.use_transfer = config.pre_att_id != ''
        self.condition = config.condition
        self.att_cnt = config.att_cnt
        self.gate_fuse = config.gate_fuse
        
        if self.uniform_graph:
            raise NotImplementedError
        if self.gate_fuse != 3:
            raise NotImplementedError
        if not self.use_transfer:
            raise NotImplementedError
        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden0, 1, True, True, 1-config.keep_prob0, False)

        cur_dim = config.hidden0 * 2
        self.gate_layer = GateLayer(cur_dim * (self.att_cnt * 2 + 1), cur_dim)
        self.scales = nn.Parameter(torch.Tensor(2, self.att_cnt).fill_(1.0))
        if not self.condition:
            self.mixt_logits_fw = nn.Parameter(torch.Tensor(self.att_cnt, config.num_mixt).zero_())
            self.mixt_logits_bw = nn.Parameter(torch.Tensor(self.att_cnt, config.num_mixt).zero_())
        else:
            self.mixt_logits_fw = nn.ModuleList([nn.Sequential(
                LockedDropout(1-config.keep_prob0),
                nn.Linear(cur_dim, config.num_mixt)
            ) for _ in range(self.att_cnt)])
            self.mixt_logits_bw = nn.ModuleList([nn.Sequential(
                LockedDropout(1-config.keep_prob0),
                nn.Linear(cur_dim, config.num_mixt)
            ) for _ in range(self.att_cnt)])

        self.qc_att = BiAttention(cur_dim * 2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(cur_dim * 8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden1, 1, False, True, 1-config.keep_prob, False)
        self.self_att = BiAttention(config.hidden1*2, 1-config.keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(config.hidden1*8, config.hidden),
                nn.ReLU()
            )

        self.original_ptr = config.original_ptr
        if not self.original_ptr:
            self.rnn_start = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
            self.linear_start = nn.Linear(config.hidden*2, 1)

            self.rnn_end = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
            self.linear_end = nn.Linear(config.hidden*2, 1)
        else:
            self.rnn_3 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
            self.summarizer = Summarizer(config.hidden*2, config.hidden, 1-config.ptr_keep_prob)
            self.pointer_net = PointerNet(config.hidden*2, config.hidden*2, config.hidden, 1-config.ptr_keep_prob)

        self.criterion = nn.CrossEntropyLoss()

        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, return_yp=False, pre_att=None, pre_att_q=None, elmo=None, elmo_q=None):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)
        
        if self.use_glove and not self.use_elmo:
            context_word = self.word_emb(context_idxs)
            ques_word = self.word_emb(ques_idxs)
        elif not self.use_glove and self.use_elmo:
            elmo_mixt_weights = F.softmax(self.elmo_mixt_logits, dim=0)
            context_word = torch.matmul(elmo.permute(0, 2, 3, 1).contiguous(), elmo_mixt_weights) * self.elmo_scale
            ques_word = torch.matmul(elmo_q.permute(0, 2, 3, 1).contiguous(), elmo_mixt_weights) * self.elmo_scale
        else: # both glove and elmo
            context_word_glove = self.word_emb(context_idxs)
            ques_word_glove = self.word_emb(ques_idxs)
            elmo_mixt_weights = F.softmax(self.elmo_mixt_logits, dim=0)
            context_word_elmo = torch.matmul(elmo.permute(0, 2, 3, 1).contiguous(), elmo_mixt_weights) * self.elmo_scale
            ques_word_elmo = torch.matmul(elmo_q.permute(0, 2, 3, 1).contiguous(), elmo_mixt_weights) * self.elmo_scale
            context_word = torch.cat([context_word_glove, context_word_elmo], dim=-1)
            ques_word = torch.cat([ques_word_glove, ques_word_elmo], dim=-1)

        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output)

        states_p, states_q = [context_output], [ques_output]

        graph_layers = pre_att.size(1) // 2
   
        def _resize(x, input_len):
            return x.permute(0, 2, 3, 1).contiguous().view(bsz, input_len, input_len, -1)
        for i in range(self.att_cnt):
            if not self.condition:
                mixt_weights_fw = F.softmax(self.mixt_logits_fw[i], dim=-1)
                mixt_weights_bw = F.softmax(self.mixt_logits_bw[i], dim=-1)
                mixt_weights_q_fw = mixt_weights_fw
                mixt_weights_q_bw = mixt_weights_bw
            else:
                mixt_weights_fw = F.softmax(self.mixt_logits_fw[i](context_output), dim=-1)
                mixt_weights_bw = F.softmax(self.mixt_logits_bw[i](context_output), dim=-1)
                mixt_weights_q_fw = F.softmax(self.mixt_logits_fw[i](ques_output), dim=-1)
                mixt_weights_q_bw = F.softmax(self.mixt_logits_bw[i](ques_output), dim=-1)

            pre_att_fw = pre_att[:, :graph_layers]
            pre_att_bw = pre_att[:, graph_layers:]
            pre_att_q_fw = pre_att_q[:, :graph_layers]
            pre_att_q_bw = pre_att_q[:, graph_layers:]
            pre_att_t_fw, pre_att_t_bw, pre_att_q_t_fw, pre_att_q_t_bw = [], [], [], []
            for l in range(graph_layers):
                pre_att_t_fw.append(pre_att_fw[:, l])
                pre_att_t_bw.append(pre_att_bw[:, l])
                pre_att_q_t_fw.append(pre_att_q_fw[:, l])
                pre_att_q_t_bw.append(pre_att_q_bw[:, l])
                if l > 0:
                    pre_att_t_fw.append(torch.matmul(pre_att_fw[:, l], pre_att_t_fw[-2]))
                    pre_att_t_bw.append(torch.matmul(pre_att_bw[:, l], pre_att_t_bw[-2]))
                    pre_att_q_t_fw.append(torch.matmul(pre_att_q_fw[:, l], pre_att_q_t_fw[-2]))
                    pre_att_q_t_bw.append(torch.matmul(pre_att_q_bw[:, l], pre_att_q_t_bw[-2]))

            pre_att_t_fw = _resize(torch.cat(pre_att_t_fw, dim=1), para_size)
            pre_att_t_bw = _resize(torch.cat(pre_att_t_bw, dim=1), para_size)
            pre_att_q_t_fw = _resize(torch.cat(pre_att_q_t_fw, dim=1), ques_size)
            pre_att_q_t_bw = _resize(torch.cat(pre_att_q_t_bw, dim=1), ques_size)

            pre_att_t_fw = torch.matmul(pre_att_t_fw, mixt_weights_fw.unsqueeze(-1)).squeeze(-1)
            pre_att_t_bw = torch.matmul(pre_att_t_bw, mixt_weights_bw.unsqueeze(-1)).squeeze(-1)
            pre_att_q_t_fw = torch.matmul(pre_att_q_t_fw, mixt_weights_q_fw.unsqueeze(-1)).squeeze(-1)
            pre_att_q_t_bw = torch.matmul(pre_att_q_t_bw, mixt_weights_q_bw.unsqueeze(-1)).squeeze(-1)

            pre_att_t_fw = torch.matmul(pre_att_t_fw, context_output)
            pre_att_t_bw = torch.matmul(pre_att_t_bw, context_output)
            pre_att_q_t_fw = torch.matmul(pre_att_q_t_fw, ques_output)
            pre_att_q_t_bw = torch.matmul(pre_att_q_t_bw, ques_output)

            states_p.extend([pre_att_t_fw * self.scales[0, i], pre_att_t_bw * self.scales[1, i]])
            states_q.extend([pre_att_q_t_fw * self.scales[0, i], pre_att_q_t_bw * self.scales[1, i]])

        context_merge = torch.cat(states_p, dim=2)
        ques_merge = torch.cat(states_q, dim=2)
        context_att = self.gate_layer(self.dropout(context_merge))
        ques_att = self.gate_layer(self.dropout(ques_merge))
        context_output = torch.cat([context_output, context_att], dim=2)
        ques_output = torch.cat([ques_output, ques_att], dim=2)

        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        output_t = self.rnn_2(output, context_lens)
        output_t = self.self_att(output_t, output_t, context_mask)
        output_t = self.linear_2(output_t)

        output = output + output_t

        if not self.original_ptr:
            output_start = self.rnn_start(output, context_lens)
            logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
            output_end = torch.cat([output, output_start], dim=2)
            output_end = self.rnn_end(output_end, context_lens)
            logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)
        else:
            output = self.rnn_3(output, context_lens)
            ques_summ = self.summarizer(ques_output[:,:, -2*self.config.hidden:], ques_mask)
            logit1, logit2 = self.pointer_net(output, ques_summ, context_mask)

        if not return_yp: return logit1, logit2

        outer = logit1[:,:,None] + logit2[:,None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return logit1, logit2, yp1, yp2

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

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

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

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
                LockedDropout(dropout),
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
            )
        # self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.net[1].weight.data)
        init.xavier_uniform(self.net[3].weight.data)
        self.net[1].bias.data.zero_()

    def forward(self, input, mask):
        att = self.net(input)
        att = att - 1e30 * (1 - mask[:,:,None])
        att = F.softmax(att, dim=1).expand_as(input)
        return (input * att).sum(1)

class Pointer(nn.Module):
    def __init__(self, input_size, state_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size+state_size, hidden_size, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        # self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.net[0].weight.data)
        init.xavier_uniform(self.net[2].weight.data)

    def forward(self, input, state, mask):
        state = state[:,None].expand_as(input)
        output = torch.cat([state, input], dim=2)
        att = self.net(output)
        logit = att - 1e30 * (1 - mask[:,:,None])
        att = F.softmax(logit, dim=1).expand_as(input)
        return (input * att).sum(1), logit.squeeze(-1)

class PointerNet(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, dropout):
        super().__init__()
        self.dropout_p = dropout
        self.dropout = LockedDropout(dropout)
        self.normal_dropout = nn.Dropout(dropout)
        self.pointer = Pointer(input_size, state_size, hidden_size)
        self.grucell = nn.GRUCell(input_size, state_size)

        # self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.grucell.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(p.data)
            else:
                p.data.zero_()

    def get_dropout_mask(self, x, dropout):
        m = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        return mask

    def forward(self, input, init, mask):
        input = self.dropout(input)
        dropout_mask = self.get_dropout_mask(init, self.dropout_p)
        output, logit1 = self.pointer(input, init * dropout_mask, mask)
        output = self.normal_dropout(output)
        state = self.grucell(output, init)
        _, logit2 = self.pointer(input, state * dropout_mask, mask)
        return logit1, logit2

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

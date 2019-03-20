import sys
import pickle
import torch
import random
import numpy

class Vocab(object):
    def __init__(self, vocab_filename, ntokens, unk, other_specials=[]):
        token2cnt = pickle.load(open(vocab_filename, 'rb'))
        token2cnt = [(k, v) for k, v in token2cnt.items()]
        token2cnt.sort(key=lambda x: x[1], reverse = True)

        self.index2token = [k for k, v in token2cnt[:ntokens]]
        self.index2token.extend([unk] + other_specials)
        self.token2index = {}
        for i, token in enumerate(self.index2token):
            self.token2index[token] = i
        self.unk = unk

    def get_index(self, token):
        if token in self.token2index:
            return self.token2index[token]
        return self.token2index[self.unk]

    def get_token(self, index):
        return self.index2token[index]

    def parse_file(self, filename):
        ret = []
        for line in open(filename, encoding='utf-8'):
            inputs = line.strip().split()
            t = torch.LongTensor(len(inputs))
            for i, token in enumerate(inputs):
                t[i] = self.get_index(token)
            ret.append(t)
        return ret

    def get_sent(self, source):
        # source: 1d Tensor

        ret = []
        for i in range(source.size(0)):
            ret.append(self.get_token(source[i]))
        return ret
    
    def __len__(self):
        return len(self.index2token)

class TensorIterator(object):
    def __init__(self, tensors, bsz, bptt, shift):
        self.tensors = tensors
        self.bsz = bsz
        self.bptt = bptt
        self.shift = shift

        self.i, self.j = 0, 0
        self.forward = True

    def get_next_batch(self):
        while self.i < len(self.tensors) and self.j + self.bptt > self.tensors[self.i].size(0):
            self.i += 1
        if self.i >= len(self.tensors): return None
        source = self.tensors[self.i][self.j: self.j+self.bptt]
        self.j += self.shift
        return source

    def __iter__(self):
        source = torch.LongTensor(self.bsz, self.bptt).cuda()
        while True:
            for i in range(self.bsz):
                source_t = self.get_next_batch()
                if source_t is None: return
                source[i].copy_(source_t)
            yield source

class FileIterator(object):
    def __init__(self, vocab, filenames, bsz, bptt, shift):
        self.vocab = vocab
        self.filenames = filenames
        random.shuffle(self.filenames)
        self.bsz = bsz
        self.bptt = bptt
        self.shift = shift

        self.i, self.j = 0, 0
        self.cur_stream = vocab.parse_file(self.filenames[0])
        self.tensors = {}
        self.forward = True

    def get_next_batch(self, i, bptt, shift):
        while i not in self.tensors or len(self.tensors[i]) < bptt:
            self.tensors[i] = self.get_next_tensor()
            if self.tensors[i] is None: return None
        source = self.tensors[i][: bptt]
        if shift >= self.tensors[i].size(0):
            shift = -1
        self.tensors[i] = self.tensors[i][shift:]
        return source

    def get_next_tensor(self):
        while self.j >= len(self.cur_stream):
            self.i += 1
            if self.i >= len(self.filenames): return None
            self.cur_stream = self.vocab.parse_file(self.filenames[self.i])
            self.j = 0
        ret = self.cur_stream[self.j]
        self.j += 1
        return ret

    def __iter__(self):
        while True:
            bptt = int(self.bptt * pow(1.4, random.uniform(-2.0, 0.0)))
            shift = self.shift + random.randint(-10, 10)
            bsz = self.bsz
            source = torch.LongTensor(bsz, bptt).cuda()
            for i in range(bsz):
                source_t = self.get_next_batch(i, bptt, shift)
                if source_t is None: return
                source[i].copy_(source_t)
            yield source


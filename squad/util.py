import torch
import numpy as np
import re
from collections import Counter
import string
import pickle
import random
from torch.autograd import Variable
import copy

RE_D = re.compile('\d')
def has_digit(string):
    return RE_D.search(string)

def prepro(token):
    return token if not has_digit(token) else 'N'

class DataIterator(object):
    def __init__(self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, pre_att_data, config, ee, idx2word_dict, tag):
        self.buckets = buckets
        self.bsz = bsz
        self.para_limit = para_limit
        self.ques_limit = ques_limit
        self.char_limit = char_limit

        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.shuffle = shuffle
        self.pre_att_data = pre_att_data

        self.ee = ee
        self.use_elmo = config.use_elmo
        self.load_elmo = config.load_elmo
        self.idx2word_dict = idx2word_dict
        self.tag = tag
        self.elmo_dim = config.elmo_dim

    #@profile
    def __iter__(self):
        context_idxs = torch.LongTensor(self.bsz, self.para_limit).cuda()
        ques_idxs = torch.LongTensor(self.bsz, self.ques_limit).cuda()
        context_char_idxs = torch.LongTensor(self.bsz, self.para_limit, self.char_limit).cuda()
        ques_char_idxs = torch.LongTensor(self.bsz, self.ques_limit, self.char_limit).cuda()
        y1 = torch.LongTensor(self.bsz).cuda()
        y2 = torch.LongTensor(self.bsz).cuda()

        if self.pre_att_data is not None:
            aux_context_idxs = torch.LongTensor(self.bsz, self.para_limit).zero_()
            aux_context_mask = torch.Tensor(self.bsz, self.para_limit).zero_()
            aux_ques_idxs = torch.LongTensor(self.bsz, self.ques_limit).zero_()
            aux_ques_mask = torch.Tensor(self.bsz, self.ques_limit).zero_()
            model = self.pre_att_data['model']
            vocab = self.pre_att_data['vocab']

        if self.use_elmo and not self.load_elmo:
            elmo_holder = torch.Tensor(self.bsz, 3, self.para_limit, self.elmo_dim).cuda().zero_()
            elmo_q_holder = torch.Tensor(self.bsz, 3, self.ques_limit, self.elmo_dim).cuda().zero_()

        idx2word_dict = self.idx2word_dict

        while True:
            if len(self.bkt_pool) == 0: break
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            ids = []

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

            if self.pre_att_data is not None:
                aux_context_mask.zero_()
                aux_ques_mask.zero_()

            if self.use_elmo and self.load_elmo:
                tokens_p, tokens_q = [], []

            for i in range(len(cur_batch)):
                context_idxs[i].copy_(cur_batch[i]['context_idxs'])
                ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
                context_char_idxs[i].copy_(cur_batch[i]['context_char_idxs'])
                ques_char_idxs[i].copy_(cur_batch[i]['ques_char_idxs'])
                y1[i] = cur_batch[i]['y1']
                y2[i] = cur_batch[i]['y2']
                ids.append(cur_batch[i]['id'])

                if self.pre_att_data is not None or self.use_elmo and self.load_elmo:
                    cur_context_idxs = cur_batch[i]['context_idxs']
                    t_tokens_p, t_tokens_q = [], []

                    for k in range(cur_context_idxs.size(0)):
                        if cur_context_idxs[k] == 0: break
                        cur_txt = idx2word_dict[str(cur_context_idxs[k])]
                        if self.pre_att_data is not None:
                            aux_context_idxs[i, k] = vocab.get_index(prepro(cur_txt))
                            aux_context_mask[i, k] = 1
                        if self.use_elmo and self.load_elmo:
                            t_tokens_p.append(cur_txt)
                    
                    cur_ques_idxs = cur_batch[i]['ques_idxs']
                    for k in range(cur_ques_idxs.size(0)):
                        if cur_ques_idxs[k] == 0: break
                        cur_txt = idx2word_dict[str(cur_ques_idxs[k])]
                        if self.pre_att_data is not None:
                            aux_ques_idxs[i, k] = vocab.get_index(prepro(cur_txt))
                            aux_ques_mask[i, k] = 1
                        if self.use_elmo and self.load_elmo:
                            t_tokens_q.append(cur_txt)

                    if self.use_elmo and self.load_elmo:
                        tokens_p.append(t_tokens_p)
                        tokens_q.append(t_tokens_q)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_q_len = int((ques_idxs[:cur_bsz] > 0).long().sum(dim=1).max())

            if self.pre_att_data is not None:
                graph_input = Variable(aux_context_idxs[:cur_bsz, :max_c_len].cuda(), volatile=True)
                mask_input = Variable(aux_context_mask[:cur_bsz, :max_c_len].cuda(), volatile=True)
                graph = model(graph_input, mask_input, return_graphs=True) # [B, LY, K, S, S]

                '''
                from torch.nn import functional as F
                import os, shutil
                if os.path.exists('vis'):
                    shutil.rmtree('vis')
                os.makedirs('vis')
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import pyplot as plt
                graph = graph / graph.max(-1, keepdim=True)[0]
                for l in range(graph.size(1)):
                    for i in range(3):
                        for k in range(1):
                            print('plotting', i, l, k)
                            plt.clf()
                            f = plt.figure(figsize=(42, 42))
                            ax = f.add_subplot(1, 1, 1)
                            to_plot = graph[i,l,k].data.cpu()
                            ax.imshow(to_plot.numpy(), cmap='hot')
                            text = [idx2word_dict[str(context_idxs[i, j])] for j in range(max_c_len)]
                            ax.set_yticks(range(len(text)))
                            ax.set_yticklabels(text, fontsize=12)
                            ax.set_xticks(range(len(text)))
                            ax.set_xticklabels(text, rotation=90, fontsize=12)
                            f.savefig('vis/i{}_l{}_k{}.png'.format(i, l, k))
                            plt.close()
                quit()
                '''
                
                ## TODO: add sorted input lengths when rnn is used in SA
                graph_q_input = Variable(aux_ques_idxs[:cur_bsz, :max_q_len].cuda(), volatile=True)
                mask_q_input = Variable(aux_ques_mask[:cur_bsz, :max_q_len].cuda())
                graph_q = model(graph_q_input, mask_q_input, return_graphs=True) # [B, LY, K, S, S]
            else:
                graph = graph_q = None

            if self.use_elmo:
                if self.load_elmo:
                    elmo, _ = self.ee.batch_to_embeddings(tokens_p)
                    elmo_q, _ = self.ee.batch_to_embeddings(tokens_q)
                else:
                    elmo_holder.zero_()
                    elmo_q_holder.zero_()
                    for i in range(len(cur_batch)):
                        cur_t = torch.load('data/elmo/{}_{}_P.pt'.format(self.tag, cur_batch[i]['id']))
                        elmo_holder[i, :, :cur_t.size(1)].copy_(cur_t)
                        cur_t = torch.load('data/elmo/{}_{}_Q.pt'.format(self.tag, cur_batch[i]['id']))
                        elmo_q_holder[i, :, :cur_t.size(1)].copy_(cur_t)
                    elmo = Variable(elmo_holder[:cur_bsz, :, :max_c_len])
                    elmo_q = Variable(elmo_q_holder[:cur_bsz, :, :max_q_len])
            else:
                elmo = elmo_q = None

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_char_idxs': ques_char_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_lens': input_lengths,
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'graph': graph if graph is not None else None,
                'graph_q': graph_q if graph_q is not None else None,
                'elmo': elmo,
                'elmo_q': elmo_q}

def get_buckets(record_file, config, limit=False):
    bucket_range = [num for num in range(*config.bucket_range)] + [1000000]
    N = len(bucket_range)
    data_buckets = [[] for _ in range(N)]

    datapoints = pickle.load(open(record_file, 'rb'))
    if config.train_limit > 0 and limit:
        cut_point = max(int(len(datapoints) * config.train_limit), 1)
        datapoints = datapoints[:cut_point]
        print('sample', len(datapoints))
    if config.is_bucket:
        for datapoint in datapoints:
            clen = int((datapoint['context_idxs'] > 0).long().sum())
            for i, br in enumerate(bucket_range):
                if clen <= br:
                    data_buckets[i].append(datapoint)
                    break
        return data_buckets
    else:
        return [datapoints]

def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

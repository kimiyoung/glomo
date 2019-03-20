import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
import pickle
from allennlp.commands.elmo import ElmoEmbedder
import torch
import os

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    idx2token_dict = {idx: token for token, idx in token2idx_dict.items()}

    return emb_mat, token2idx_dict, idx2token_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    datapoints = []
    total = 0
    total_ = 0
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1

        context_idxs = torch.LongTensor(para_limit).zero_()
        context_char_idxs = torch.LongTensor(para_limit, char_limit).zero_()
        ques_idxs = torch.LongTensor(ques_limit).zero_()
        ques_char_idxs = torch.LongTensor(ques_limit, char_limit).zero_()

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1, y2 = start, end

        datapoints.append({'context_idxs': context_idxs,
            'context_char_idxs': context_char_idxs,
            'ques_idxs': ques_idxs,
            'ques_char_idxs': ques_char_idxs,
            'y1': y1,
            'y2': y2,
            'id': example['id']})
    print("Build {} / {} instances of features in total".format(total, total_))
    pickle.dump(datapoints, open(out_file, 'wb'), protocol=-1)

def extract_elmo_file(config, record_file, idx2word_dict, ee, tag):
    if not os.path.exists('data/elmo'):
        os.makedirs('data/elmo')
    print('loading datapoints...')
    datapoints = pickle.load(open(record_file, 'rb'))
    i = 0
    cnt = 0
    BSZ, N = 32, len(datapoints)
    while i < N:
        print('extracting ELMO batch #{}/{} {}'.format(cnt, N//BSZ, record_file))
        j = min(i + BSZ, N)
        for type in ['P', 'Q']:
            cur_tokens = []
            for k in range(i, j):
                t_tokens = []
                cur_context_idxs = datapoints[k]['context_idxs'] if type == 'P' else datapoints[k]['ques_idxs']
                for l in range(cur_context_idxs.size(0)):
                    if cur_context_idxs[l] == 0: break
                    t_tokens.append(idx2word_dict[str(cur_context_idxs[l])])
                cur_tokens.append(t_tokens)
            embeddings, _ = ee.batch_to_embeddings(cur_tokens)
            embeddings = embeddings.data.cpu()
            for num, k in enumerate(range(i, j)):
                torch.save(embeddings[num, :, :len(cur_tokens[num])].contiguous(), 'data/elmo/{}_{}_{}.pt'.format(tag, datapoints[k]['id'], type))
        i = j
        cnt += 1

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)

def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word2idx_dict = None
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    word_emb_mat, word2idx_dict, idx2word_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim, token2idx_dict=word2idx_dict)

    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    char_emb_mat, char2idx_dict, idx2char_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim, token2idx_dict=char2idx_dict)

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.char2idx_file, char2idx_dict, message="char2idx")
    save(config.idx2word_file, idx2word_dict, message='idx2word')
    save(config.idx2char_file, idx2char_dict, message='idx2char')

def extract_elmo(config):
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)
    print('constructing embedder...')
    # ee = ElmoEmbedder(cuda_device=0)
    # torch.save(ee, config.elmo_ee_file)
    ee = torch.load(config.elmo_ee_file)
    extract_elmo_file(config, config.dev_record_file, idx2word_dict, ee, 'dev')
    #extract_elmo_file(config, config.train_record_file, idx2word_dict, ee, 'train')

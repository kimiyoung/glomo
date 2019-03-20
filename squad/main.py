import os
from prepro import prepro, extract_elmo
from run import train, debug
import argparse

parser = argparse.ArgumentParser()

train_file = os.path.join("data", "train-v1.1.json")
dev_file = os.path.join("data", "dev-v1.1.json")
test_file = os.path.join("data", "dev-v1.1.json")
glove_word_file = os.path.join("data", "glove.840B.300d.txt")

target_dir = "data"
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")
idx2word_file = os.path.join(target_dir, 'idx2word.json')
idx2char_file = os.path.join(target_dir, 'idx2char.json')
train_record_file = os.path.join(target_dir, 'train_record.pkl')
dev_record_file = os.path.join(target_dir, 'dev_record.pkl')
test_record_file = os.path.join(target_dir, 'test_record.pkl')

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--target_dir', type=str, default=target_dir)
parser.add_argument('--train_file', type=str, default=train_file)
parser.add_argument('--dev_file', type=str, default=dev_file)
parser.add_argument('--test_file', type=str, default=test_file)
parser.add_argument('--glove_word_file', type=str, default=glove_word_file)
parser.add_argument('--save', type=str, default='data/QA')

parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
parser.add_argument('--char_emb_file', type=str, default=char_emb_file)
parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
parser.add_argument('--char2idx_file', type=str, default=char2idx_file)
parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
parser.add_argument('--idx2char_file', type=str, default=idx2char_file)

parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--glove_char_size', type=int, default=94)
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=8)

parser.add_argument('--para_limit', type=int, default=400)
parser.add_argument('--ques_limit', type=int, default=50)
parser.add_argument('--test_para_limit', type=int, default=1000)
parser.add_argument('--test_ques_limit', type=int, default=100)
parser.add_argument('--char_limit', type=int, default=16)
parser.add_argument('--word_count_limit', type=int, default=-1)
parser.add_argument('--char_count_limit', type=int, default=-1)

parser.add_argument('--capacity', type=int, default=15000)
parser.add_argument('--num_threads', type=int, default=4)
parser.add_argument('--not_use_cudnn', dest='use_cudnn', action='store_false', default=True)
parser.add_argument('--no_bucket', dest='is_bucket', action='store_false', default=True)
parser.add_argument('--bucket_range', nargs='+', default=[40, 361, 40])

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_steps', type=int, default=60000)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--val_num_batches', type=int, default=150)
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--keep_prob', type=float, default=0.7)
parser.add_argument('--ptr_keep_prob', type=float, default=0.7)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--hidden', type=int, default=76)
parser.add_argument('--char_hidden', type=int, default=100)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--att_K', type=int, default=1)

parser.add_argument('--pretrained_char', action='store_true')
parser.add_argument('--fasttext', action='store_true')

parser.add_argument('--pre_att_id', type=str, default='')
parser.add_argument('--pre_att_id1', type=str, default='')
parser.add_argument('--pre_att_id2', type=str, default='')
parser.add_argument('--pre_att_id3', type=str, default='')
parser.add_argument('--pre_att_id4', type=str, default='')
parser.add_argument('--pre_att_id5', type=float, default=1.0)

parser.add_argument('--hidden0', type=int, default=76)
parser.add_argument('--hidden1', type=int, default=76)
parser.add_argument('--keep_prob0', type=float)
parser.add_argument('--train_limit', type=float, default=0)
# parser.add_argument('--transfer_graph', action='store_true')
parser.add_argument('--elmo_dim', type=int, default=1024)
parser.add_argument('--use_glove', action='store_true')
parser.add_argument('--use_elmo', action='store_true')
parser.add_argument('--elmo_ee_file', type=str, default=os.path.join(target_dir, 'elmo_ee.pt'))
parser.add_argument('--load_elmo', action='store_true')
parser.add_argument('--condition', type=int, default=0)
parser.add_argument('--original_ptr', action='store_true')
parser.add_argument('--att_cnt', type=int, default=1)
parser.add_argument('--gate_fuse', type=int, default=0)
parser.add_argument('--train_emb', type=int, default=0)
parser.add_argument('--optim', type=str, default='adadelta')
parser.add_argument('--uniform_graph', type=int, default=0)
parser.add_argument('--trnn', action='store_true')
parser.add_argument('--square', action='store_true')

config = parser.parse_args()
config.condition = config.condition > 0
config.train_emb = config.train_emb > 0
config.uniform_graph = config.uniform_graph > 0
if config.pre_att_id == '' and config.pre_att_id1 != '':
    config.pre_att_id = 'SKIP-v16-cnn-gru-yann-ly{}-k{}q{}v1-lr8.0-sp{}-topk{}'.format(config.pre_att_id1, config.pre_att_id2, config.pre_att_id3, config.pre_att_id4, config.pre_att_id5)

if config.mode == 'train':
    train(config)
elif config.mode == 'prepro':
    prepro(config)
elif config.mode == 'debug':
    debug(config)
elif config.mode == 'elmo':
    extract_elmo(config)

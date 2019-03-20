from data import Vocab, TensorIterator, FileIterator
import argparse
import random
import numpy as np
import torch
import time
import os, shutil
from model import Model
from torch.autograd import Variable
import pickle
import math
from torch import optim

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

parser = argparse.ArgumentParser()
parser.add_argument('--bsz', type=int, default=32, help='batch size')
parser.add_argument('--bptt', type=int, default=200, help='backprop TT length')
parser.add_argument('--dec_len', type=int, default=3, help='decoder length')
parser.add_argument('--shift', type=int, default=100, help='shift between batches')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--save', type=str, default='SKIP', help='save name')
parser.add_argument('--d', type=int, default=512, help='dimension')
parser.add_argument('--ntokens', type=int, default=100000, help='vocab size')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--nlayers_key', type=int, default=1, help='number of layers for conv if needed.')
parser.add_argument('--nlayers_query', type=int, default=1, help='number of layers for conv if needed.')
parser.add_argument('--use_value', type=int, default=1, help='number of layers for conv if needed.')
parser.add_argument('--dec_layers', type=int, default=2, help='number of decoder layers')
parser.add_argument('--lr', type=float, default=8.0, help='learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--eval-interval', type=int, default=4000, help='evaluation interval')
parser.add_argument('--log-interval', type=int, default=200, help='logging interval')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.2)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--decay_rate', type=float, default=2.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--sample_len', type=int, default=20)
parser.add_argument('--kernel_size', type=int, default=3)
#parser.add_argument('--no_gru', dest='use_gru', action='store_false', default=True)
#parser.add_argument('--cnn_sa', action='store_true')
parser.add_argument('--no_gru', type=int, default=0)
parser.add_argument('--cnn_sa', type=int, default=0)
parser.add_argument('--yann', type=int, default=0)
parser.add_argument('--sparse_mode', type=int, default=0)
parser.add_argument('--topk_rate', type=float, default=1.0) # actual top k = topk_rate * sqrt(seq_len)
args = parser.parse_args()
args.use_gru = args.no_gru == 0
args.cnn_sa = args.cnn_sa > 0
args.yann = args.yann > 0

if args.use_gru:
    assert(args.use_value==1)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = 'SKIP-v16-{}-{}-{}-ly{}-k{}q{}v{}-lr{}-sp{}-topk{}-declen{}'.format('cnn' if args.cnn_sa else 'ff', 'gru' if args.use_gru else 'add', 'softmax' if not args.yann else 'yann', args.nlayers, args.nlayers_key, args.nlayers_query, args.use_value, args.lr, args.sparse_mode, args.topk_rate, args.dec_len)
create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py', 'nn_utils.py'])
def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

logging('Args')
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))

vocab = Vocab('vocabv2.pkl', args.ntokens, '<unk>')

def get_file_list(filename):
    return [line.strip() for line in open(filename)]

def get_tensors(filenames, cache_file=None):
    if cache_file is not None and os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))
    ret = []
    for filename in filenames:
        ret.extend(vocab.parse_file(filename))
    if cache_file is not None:
        pickle.dump(ret, open(cache_file, 'wb'))
    return ret

train_files = get_file_list('trainv2.txt')
valid_files = get_file_list('validv2.txt')
test_files = get_file_list('testv2.txt')

valid_tensors = get_tensors(valid_files, 'valid_cachev2.pkl')
test_tensors = get_tensors(test_files, 'test_cachev2.pkl')

def build_valid_iter():
    return TensorIterator(valid_tensors, args.bsz, args.bptt, args.shift)

def build_test_iter():
    return TensorIterator(test_tensors, args.bsz, args.bptt, args.shift)

def build_train_iter():
    return FileIterator(vocab, train_files, args.bsz, args.bptt, args.shift)

model = Model(args.d, len(vocab), args.nlayers, args.nlayers_key, args.nlayers_query, args.use_value, args.dec_layers, args.dropout, args.dec_len, args.sample_len, args.kernel_size, args.use_gru, args.cnn_sa, args.yann, args.sparse_mode, args.topk_rate)
logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
model = model.cuda()

lr = args.lr
best_val_loss = None
end_of_train = False

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

def evaluate(data_source):
    model.eval()
    total_loss, total_len = 0, 0
    for source in data_source:
        source = Variable(source, volatile=True)

        cur_loss = model(source)
        cur_len = source.size(0) * source.size(1)
        total_loss += cur_loss.data[0] * cur_len
        total_len += cur_len
    return total_loss / total_len

def train():
    global lr
    global best_val_loss
    global end_of_train

    model.train()
    total_loss = 0
    cur_patience = 0
    start_time = time.time()
    eval_start_time = time.time()

    data_source = build_train_iter()
    for batch, source in enumerate(data_source):
        source = Variable(source)

        optimizer.zero_grad()
        loss = model(source)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data[0]

        # Logging every args.log_interval batches
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d} batches | lr {:03.3f} | ms/batch {:5.2f} | '
                    'train ppl {:8.2f}'.format(epoch, batch, lr, elapsed * 1000 / args.log_interval, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # Evaluate every args.eval_interval batches
        if batch % args.eval_interval == 0 and batch > 0:
            valid_loss = evaluate(build_valid_iter())
            logging('-' * 89)
            logging('| eval {:3d} in epoch {:3d} | time: {:5.2f}s | '
                    'valid ppl {:8.2f}'.format(batch // args.eval_interval, epoch, 
                    time.time() - eval_start_time, math.exp(valid_loss)))
            logging('-' * 89)
            eval_start_time = time.time()
            # Save the model if the validation loss is the best we've seen so far.
            if best_val_loss is None or valid_loss < best_val_loss:
                torch.save(model, os.path.join(args.save, 'model.pt'))
                # if hasattr(model, 'st_predictor'):
                #     torch.save(model.st_predictor.state_dict(), os.path.join(args.save, 'st_predictor.pt'))
                best_val_loss = valid_loss
                cur_patience = 0
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                cur_patience += 1
                if cur_patience >= args.patience:
                    lr /= args.decay_rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    if lr < args.lr * 1e-2:
                        end_of_train = True
                        break
                    cur_patience = 0
            model.train()

try:
    for epoch in range(1, args.epochs):
        train()
        if end_of_train:
            break
except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model.pt'))

# Run on test data.
test_loss = evaluate(build_test_iter())
logging('=' * 89)
logging('| End of training | test ppl {:8.2f}'.format(
        math.exp(test_loss)))
logging('=' * 89)



import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model_t import Model
from model_trnn import ModelTRNN
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F

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

criterion = nn.CrossEntropyLoss()

def print_scale(config):
    with open(config.idx2word_file, 'r') as fh:
        idx2word = json.load(fh)
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    state_dict = torch.load('data/QA-20180415-224240/model.pt')
    model = Model(config, word_mat, char_mat).cuda()
    model.load_state_dict(state_dict)
    print('scale', model.scale)

def debug(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    print_scale(config)

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.train_eval_file, "r") as fh:
    #     train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.pre_att_id != '':
        config.save = 'T16-v2-{}-kp0{}-cond{}-ori{}-attcnt{}-gatefuse{}-lr{}-opt{}'.format(config.pre_att_id, config.keep_prob0, int(config.condition), int(config.original_ptr), config.att_cnt, config.gate_fuse, config.init_lr, config.optim)
        if config.use_elmo:
            config.save += "_ELMO"
        if config.train_emb:
            raise ValueError
            config.save += "_TE"
        if config.trnn:
            config.save += '_TRNN'
    else:
        config.save = 'baseline-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
        if config.use_elmo:
            config.save += "_ELMO"
        if config.uniform_graph:
            config.save += '_UNIFORM'
    # non overwriting
    # if os.path.exists(config.save):
    #     sys.exit(1)
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'main.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.pre_att_id != '':
        sys.path.insert(0, '../pretrain')
        from data import Vocab
        vocab = Vocab('../pretrain/vocabv2.pkl', 100000, '<unk>')
        
        # from model8 import StructurePredictor
        # model = StructurePredictor(512, len(vocab), 1, 1, 0.0)
        # model.load_state_dict(torch.load('../skip_thought/{}/st_predictor.pt'.format(config.pre_att_id)))
        # model.cuda()
        # model.eval()

        model = torch.load('../pretrain/{}/model.pt'.format(config.pre_att_id))
        # if 'gru' in config.pre_att_id:
        #     model.set_gru(True)
        # elif 'add' in config.pre_att_id:
        #     model.set_gru(False)
        # else:
        #     assert False
        model.cuda()
        ori_model = model
        model = nn.DataParallel(model)
        model.eval()
        import re
        try:
            nly = int(re.search(r'ly(\d+)', config.pre_att_id).group(1))
        except:
            nly = len(ori_model.enc_net.nets)
        if config.gate_fuse < 3:
            config.num_mixt = nly * 8
        else:
            config.num_mixt = (nly + nly - 1) * 8

        # old_model = torch.load('../skip_thought/{}/model.pt'.format(config.pre_att_id))
        # from model5 import GraphModel
        # model = GraphModel(old_model).cuda()
        # model = nn.DataParallel(model)
        # model.eval()
        # del old_model
        # import gc
        # gc.collect()
        # from data import Vocab
        # vocab = Vocab('../skip_thought/vocabv2.pkl', 100000, '<unk>')
        
        del sys.path[0]
        
        pre_att_data = {'model': model, 'vocab': vocab}
    else:
        pre_att_data = None

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    if config.use_elmo and config.load_elmo:
        ee = torch.load(config.elmo_ee_file)
    else:
        ee = None

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file, config, limit=True)
    dev_buckets = get_buckets(config.dev_record_file, config, limit=False)

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, pre_att_data, config, ee, idx2word_dict, 'train')

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, pre_att_data, config, ee, idx2word_dict, 'dev')

    model = Model(config, word_mat, char_mat) if not config.trnn else ModelTRNN(config, word_mat, char_mat)
    # logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    # ori_model.word_emb.cpu()
    # model = ori_model
    model = nn.DataParallel(ori_model)

    lr = config.init_lr
    # optimizer = optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.momentum)
    if config.optim == "adadelta":  # default
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr, rho=0.95)
    elif config.optim == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr, momentum=config.momentum)
    elif config.optim == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr, betas=(config.momentum, 0.999))
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    for epoch in range(10000 * 32 // config.batch_size):
        for data in build_train_iterator():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])

            graph = data['graph']
            graph_q = data['graph_q']
            if graph is not None:
                graph.volatile = False
                graph.requires_grad = False
                graph_q.volatile = False
                graph_q.requires_grad = False

            elmo, elmo_q = data['elmo'], data['elmo_q']
            if elmo is not None:
                elmo.volatile = False
                elmo.requires_grad = False
                elmo_q.volatile = False
                elmo_q.requires_grad = False

            logit1, logit2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, pre_att=graph, pre_att_q=graph_q, elmo=elmo, elmo_q=elmo_q)
            loss = criterion(logit1, y1) + criterion(logit2, y2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            import gc; gc.collect()

            total_loss += loss.data[0]
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % (config.checkpoint * 32 // config.batch_size)  == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file)
                model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                debug_s = ''
                if hasattr(ori_model, 'scales'):
                    debug_s += '| scales {} '.format(ori_model.scales.data.cpu().numpy().tolist())
                # if hasattr(ori_model, 'mixt_logits') and (not hasattr(ori_model, 'condition') or not ori_model.condition):
                #     debug_s += '| mixt {}'.format(F.softmax(ori_model.mixt_logits, dim=-1).data.cpu().numpy().tolist())
                if debug_s != '':
                    logging(debug_s)
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

def evaluate_batch(data_source, model, max_batches, eval_file):
    answer_dict = {}
    total_loss, step_cnt = 0, 0
    for step, data in enumerate(data_source):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)

        graph = data['graph']
        graph_q = data['graph_q']

        elmo = data['elmo']
        elmo_q = data['elmo_q']
        if elmo is not None:
            elmo.volatile = True
            elmo_q.volatile = True

        logit1, logit2, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, return_yp=True, pre_att=graph, pre_att_q=graph_q, elmo=elmo, elmo_q=elmo_q)
        loss = criterion(logit1, y1) + criterion(logit2, y2)
        answer_dict_, _ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist())
        answer_dict.update(answer_dict_)

        total_loss += loss.data[0]
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss
    return metrics

# def test(config):
#     with open(config.word_emb_file, "r") as fh:
#         word_mat = np.array(json.load(fh), dtype=np.float32)
#     with open(config.char_emb_file, "r") as fh:
#         char_mat = np.array(json.load(fh), dtype=np.float32)
#     with open(config.test_eval_file, "r") as fh:
#         eval_file = json.load(fh)
#     with open(config.test_meta, "r") as fh:
#         meta = json.load(fh)

#     total = meta["total"]

#     print("Loading model...")
#     test_batch = get_dataset(config.test_record_file, get_record_parser(
#         config, is_test=True), config).make_one_shot_iterator()

#     model = Model(config, test_batch, word_mat, char_mat, trainable=False)

#     sess_config = tf.ConfigProto(allow_soft_placement=True)
#     sess_config.gpu_options.allow_growth = True

#     with tf.Session(config=sess_config) as sess:
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
#         sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
#         losses = []
#         answer_dict = {}
#         remapped_dict = {}
#         for step in tqdm(range(total // config.batch_size + 1)):
#             qa_id, loss, yp1, yp2 = sess.run(
#                 [model.qa_id, model.loss, model.yp1, model.yp2])
#             answer_dict_, remapped_dict_ = convert_tokens(
#                 eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
#             answer_dict.update(answer_dict_)
#             remapped_dict.update(remapped_dict_)
#             losses.append(loss)
#         loss = np.mean(losses)
#         metrics = evaluate(eval_file, answer_dict)
#         with open(config.answer_file, "w") as fh:
#             json.dump(remapped_dict, fh)
#         print("Exact Match: {}, F1: {}".format(
#             metrics['exact_match'], metrics['f1']))

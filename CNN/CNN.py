import argparse
import torch
import os
import random
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import sklearn
from sklearn.metrics import balanced_accuracy_score
import torch.optim as optim


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'CNN'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir', type=str,
                            default='./data',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')

        # word embedding
        parser.add_argument('--embedding_path', type=str,
                            default='C:\\Users\\kevin\\OneDrive\\Documents\\MSU\\sem 3\\NLP\\Project\\step 5\\CNN\\CNN-relation-extraction\\glove-6b-60d.txt',
                            help='pre_trained word embedding')
        parser.add_argument('--word_dim', type=int,
                            default=50,
                            help='dimension of word embedding')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--mode', type=int,
                            default=1,
                            choices=[0, 1],
                            help='running mode: 1 for training; otherwise testing')
        parser.add_argument('--seed', type=int,
                            default=5782,
                            help='random seed')
        parser.add_argument('--cuda', type=int,
                            default=0,
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--epoch', type=int,
                            default=20,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--dropout', type=float,
                            default=0.5,
                            help='the possiblity of dropout')
        parser.add_argument('--batch_size', type=int,
                            default=128,
                            help='batch size')
        parser.add_argument('--lr', type=float,
                            default=0.001,
                            help='learning rate')
        parser.add_argument('--max_len', type=int,
                            default=100,
                            help='max length of sentence')
        parser.add_argument('--pos_dis', type=int, default=50,
                            help='max distance of position embedding')
        parser.add_argument('--pos_dim', type=int,
                            default=5,
                            help='dimension of position embedding')
        parser.add_argument('--hidden_size', type=int, default=100,
                            help='the size of linear layer between convolution and classification')

        # hyper parameters for cnn
        parser.add_argument('--filter_num', type=int, default=200,
                            help='the number of filters in convolution')
        parser.add_argument('--window', type=int, default=3,
                            help='the size of window in convolution')

        parser.add_argument('--L2_decay', type=float, default=1e-5,
                            help='L2 weight decay')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])




class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        pad_emb = np.zeros([1, self.word_dim], dtype=np.float32)  # <pad> is initialize as zero
        word_vec = np.concatenate((pad_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class SemEvalDateset(Dataset):
    def __init__(self, filename, rel2id, word2id, config):
        self.filename = filename
        self.rel2id = rel2id
        self.word2id = word2id
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['*UNKNOWN*']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class SemEvalDataLoader(object):
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.config = config

    def collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def get_data(self, filename, shuffle=False):
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.collate_fn  # Use the public collate_fn method
        )
        return loader

    def get_train(self):
        return self.get_data('train.json', shuffle=True)

    def get_dev(self):
        return self.get_data('test.json', shuffle=False)

    def get_test(self):
        return self.get_data('test.json', shuffle=False)



class CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window
        self.hidden_size = config.hidden_size

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_value)
        self.linear = nn.Linear(
            in_features=self.filter_num,
            out_features=self.hidden_size,
            bias=True
        )
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        init.xavier_normal_(self.pos1_embedding.weight)
        init.xavier_normal_(self.pos2_embedding.weight)
        init.xavier_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0.)
        init.xavier_normal_(self.linear.weight)
        init.constant_(self.linear.bias, 0.)
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        return emb  # B*L*D, D=word_dim+2*pos_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        emb = self.encoder_layer(token, pos1, pos2)
        emb = self.dropout(emb)
        conv = self.conv_layer(emb, mask)
        pool = self.single_maxpool_layer(conv)
        sentence_feature = self.linear(pool)
        sentence_feature = self.tanh(sentence_feature)
        sentence_feature = self.dropout(sentence_feature)
        logits = self.dense(sentence_feature)
        return logits

def semeval_scorer(predict_label, true_label, class_num=10):
    import math
    assert true_label.shape[0] == predict_label.shape[0]
    confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
    xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
    for i in range(true_label.shape[0]):
        true_idx = math.ceil(true_label[i]/2)
        predict_idx = math.ceil(predict_label[i]/2)
        if true_label[i] == predict_label[i]:
            confusion_matrix[predict_idx][true_idx] += 1
        else:
            if true_idx == predict_idx:
                xDIRx[predict_idx] += 1
            else:
                confusion_matrix[predict_idx][true_idx] += 1

    col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)
    row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)
    f1 = np.zeros(shape=[class_num], dtype=np.float32)

    for i in range(0, class_num):  # ignore the 'Other'
        try:
            p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
            r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
            f1[i] = (2 * p * r / (p + r))
        except:
            pass
    actual_class = 0
    total_f1 = 0.0
    for i in range(1, class_num):
        if f1[i] > 0.0:  # classes that not in the predict label are not considered
            actual_class += 1
            total_f1 += f1[i]
    try:
        macro_f1 = total_f1 / actual_class
    except:
        macro_f1 = 0.0
    return macro_f1


class Eval(object):
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logits = model(data)
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]

                _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        f1 = semeval_scorer(predict_label, true_label)
        balanced_accuray = balanced_accuracy_score(true_label , predict_label)
        return f1, eval_loss, predict_label, balanced_accuray


def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.L2_decay)

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    max_f1 = -float('inf')
    for epoch in range(1, config.epoch+1):
        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        _, train_loss, _ ,_= eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ , balanced_Acc = eval_tool.evaluate(model, criterion, dev_loader)
        
        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f | Balanced Accuracy : %.4f'
              % (epoch, train_loss, dev_loss, f1,balanced_Acc), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(
                config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')

    _, _, test_loader = loader
    model.load_state_dict(torch.load(
        os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)
    f1, test_loss, predict_label,bal_acc = eval_tool.evaluate(
        model, criterion, test_loader)
    print('test_loss: %.3f | micro f1 on test:  %.4f | Balanced Accuracy : %.4f' % (test_loss, f1,bal_acc))
    return predict_label

if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    print('--------------------------------------')
    model = CNN(word_vec=word_vec, class_num=class_num, config=config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()


    if config.mode == 1:  # train mode
        train(model, criterion, loader, config)
    predict_label = test(model, criterion, loader, config)
    print_result(predict_label, id2rel)

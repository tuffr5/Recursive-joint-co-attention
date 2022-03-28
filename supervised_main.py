import os
import time
import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import random

import utils
from dataset import create_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from models import DCNWithRCNN


parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AVE',
                    help='model name')
parser.add_argument('--model_dir', type=str, default='model',
                    help='model to store and test')
parser.add_argument('--dataset_mode', type=str, default='AVE',
                    help='chooses how datasets are loaded. [eventfilter for event_detection|]')
parser.add_argument('--dir_video', type=str, default="data/visual_feature.h5",
                    help='visual features')
parser.add_argument('--dir_audio', type=str,
                    default='data/audio_feature.h5',
                    help='audio features')
parser.add_argument('--dir_labels', type=str, default='data/labels.h5',
                    help='labels of AVE dataset')

parser.add_argument('--dir_order_train', type=str, default='data/train_order.h5',
                    help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='data/val_order.h5',
                    help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='data/test_order.h5',
                    help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--epoch', type=int, default=None,
                    help='epoch to test')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')

parser.add_argument('--train', action='store_true', default=False,
                    help='train a new model')
parser.add_argument('--phase', type=str, default='train',
                    help='phase of dataset [train | val | test]')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--serial_batches', action='store_true',
                    help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0')

#### model params
parser.add_argument("--rnn", type=str, default="LSTM", help='LSTM | GRU')
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--video_size", type=int, default=512)
parser.add_argument("--audio_size", type=int, default=512)
parser.add_argument("--num_seq", type=int, default=4)
parser.add_argument("--droprnn", type=float, default=0.1)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--num_classes", type=float, default=29)

opt = parser.parse_args()
opt.model_dir = './checkpoints/'+ opt.model_dir

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id  # GPU ID

val_opt = copy.deepcopy(opt)

# model
model_name = opt.model_name
# dim1, dim2, embedding_dim, target_size
net_model = DCNWithRCNN(opt)

net_model.cuda()

criterionCLF = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(net_model.parameters(), lr=opt.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)


# scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    expr_dir = os.path.join(opt.model_dir, opt.phase)
    utils.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def compute_acc(labels, x_labels):
    N = int(labels.shape[0] * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(labels.shape[0]):
        for j in range(x_labels.shape[1]):
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1

    return accuracy_score(real_labels, pre_labels)


def train(opt):
    print('The number of training samples = %d' % len(train_dataset))
    best_val_acc = 0
    for epoch in range(opt.nb_epoch):
        epoch_loss = 0
        n = 0
        acc = 0
        start = time.time()
        for i, data in enumerate(train_dataset):
            audio_inputs = Variable(data['audio'].cuda(), requires_grad=False)
            video_inputs = Variable(data['video'].cuda(), requires_grad=False)
            labels = Variable(data['label'].cuda(), requires_grad=False)
            net_model.zero_grad()
            scores = net_model(audio_inputs, video_inputs)

            loss = criterionCLF(scores, labels)

            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            labels = labels.cpu().data.numpy()
            scores = scores.cpu().data.numpy()
            acc += compute_acc(labels, scores)
            n = n + 1
            optimizer.step()
        scheduler.step(epoch_loss / n)

        end = time.time()
        print("=== Epoch {%s}   Total_Loss: {%.4f}  Acc: {%.4f}  Running time: {%.4f}"
              % (str(epoch), epoch_loss / n, acc / n, end - start))
        if epoch % 5 == 0:
            val_acc = val(val_opt)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(net_model, opt.model_dir + '/test.pt')
        if epoch % 10 == 0:
            torch.save(net_model, opt.model_dir + '/test' + str(epoch) + '.pt')


def val(val_opt):
    net_model.eval()
    n = 0
    acc = 0
    loss = 0
    for i, data in enumerate(val_dataset):
        audio_inputs = Variable(data['audio'].cuda(), requires_grad=False)
        video_inputs = Variable(data['video'].cuda(), requires_grad=False)
        labels = Variable(data['label'].cuda(), requires_grad=False)
        x_labels = net_model(audio_inputs, video_inputs)

        loss += criterionCLF(x_labels, labels)

        labels = labels.cpu().data.numpy()
        x_labels = x_labels.cpu().data.numpy()
        acc += compute_acc(labels, x_labels)
        n = i + 1
    print("=== Loss: {%.4f}  Acc: {%.4f}" % (loss / n, acc / n))
    net_model.train()
    return acc / n


def test(opt):
    if opt.epoch is not None:
        model = torch.load(opt.model_dir + '/test' + str(opt.epoch) + '.pt')
    else:
        model = torch.load(opt.model_dir + '/test.pt')
    model.eval()
    n = 0
    acc = 0
    loss = 0
    for i, data in enumerate(test_dataset):
        audio_inputs = Variable(data['audio'].cuda(), requires_grad=False)
        video_inputs = Variable(data['video'].cuda(), requires_grad=False)
        labels = Variable(data['label'].cuda(), requires_grad=False)
        x_labels = model(audio_inputs, video_inputs)

        loss += criterionCLF(x_labels, labels)

        labels = labels.cpu().data.numpy()
        x_labels = x_labels.cpu().data.numpy()
        acc += compute_acc(labels, x_labels)
        n = i + 1
    print("=== Loss: {%.4f}  Acc: {%.4f}" % (loss / n, acc / n))
    return acc / n


# training and testing
if opt.train:
    opt.batch_size = 64
    opt.phase = 'train'
    print_options(opt)
    train_dataset = create_dataset(opt)
    val_opt.batch_size = 64
    val_opt.phase = 'val'
    print_options(val_opt)
    val_dataset = create_dataset(val_opt)
    train(opt)
    random.seed(3339)
else:
    random.seed(402)
    opt.batch_size = 64
    opt.phase = 'test'
    # print_options(opt)
    test_dataset = create_dataset(opt)
    test(opt)

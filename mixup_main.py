# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import sys
import numpy as np
from random import shuffle
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type=str,
                    help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type=str,
                    help=' cifar10 or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)

# Adjust learning rate and for SGD Optimizer


def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model


def double_mixup(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(2*batch_size).cuda()
    else:
        index = torch.randperm(2*batch_size)

    x_extended = torch.cat((x, x))
    y_extended = torch.cat((y, y))

    mixed_x = lam * x_extended + (1 - lam) * x[index, :]
    y_a, y_b = y_extended, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# def average_mixup(x, y, alpha=1.0, use_cuda=True, num_classes=10):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     # if alpha > 0:
#     #     lam = np.random.beta(alpha, alpha)
#     # else:
#     #     lam = 1
#     lam = 0.9
#     # iterate through each class
#     counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     for label_class in range(num_classes):
#         # find all samples with this label class
#         indices = [index for index, value in enumerate(
#             y) if value == label_class]
#         label_samples = x[indices]
#         # calculate average image
#         avg_img = torch.mean(label_samples, 0)
#         # mix
#         x[indices] = lam * x[indices] + (1 - lam) * avg_img
#         counts[label_class] = len(indices)
#     print(counts)
#     return x, y, lam


# def smart_mixup(x, y, alpha=1.0, use_cuda=True, num_classes=10):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     batch_size = x.size()[0]
#     if use_cuda:
#         rand_index = torch.randperm(batch_size).cuda()
#     else:
#         rand_index = torch.randperm(batch_size)

#     # mixup based on similar pairs
#     index_classes = [[], [], [], [], [], [], [], [], [], []]
#     index_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     # add each index to respective class
#     for index in rand_index:
#         index_classes[y[index]].append(index)
#         index_counts[y[index]] += 1
#     index_total = index_counts.copy()
#     unused_index = []
#     need_to_set = []
#     i = 0
#     # now create new list of indexes to mix on
#     indices = []
#     for index in rand_index:
#         # airplane
#         if y[index] == 0 and index_counts[2] > 0:
#             # bird
#             indices.append(index_classes[2][index_total[2] - index_counts[2]])
#             index_counts[2] -= 1
#         # automobile
#         elif y[index] == 1 and index_counts[9] > 0:
#             # truck
#             indices.append(index_classes[9][index_total[9] - index_counts[9]])
#             index_counts[9] -= 1
#         # bird
#         elif y[index] == 2 and index_counts[0] > 0:
#             # airplane
#             indices.append(index_classes[0][index_total[0] - index_counts[0]])
#             index_counts[0] -= 1
#         # cat
#         elif y[index] == 3 and index_counts[5] > 0:
#             # dog
#             indices.append(index_classes[5][index_total[5] - index_counts[5]])
#             index_counts[5] -= 1
#         # deer
#         elif y[index] == 4 and index_counts[7] > 0:
#             # horse
#             indices.append(index_classes[7][index_total[7] - index_counts[7]])
#             index_counts[7] -= 1
#         # dog
#         elif y[index] == 5 and index_counts[3] > 0:
#             # cat
#             indices.append(index_classes[3][index_total[3] - index_counts[3]])
#             index_counts[3] -= 1
#         # horse
#         elif y[index] == 7 and index_counts[4] > 0:
#             # deer
#             indices.append(index_classes[4][index_total[4] - index_counts[4]])
#             index_counts[4] -= 1
#         # truck
#         elif y[index] == 9 and index_counts[1] > 0:
#             # automobile
#             indices.append(index_classes[1][index_total[1] - index_counts[1]])
#             index_counts[1] -= 1
#         else:
#             # add itself
#             indices.append(-1)
#             need_to_set.append(i)
#             unused_index.append(index)
#         i += 1

#     # randomly permute unused_index
#     shuffle(unused_index)
#     # set unused to normal mixup
#     indices = np.array(indices)
#     indices[need_to_set] = unused_index
#     indices = indices.tolist()
#     # print('num use:', (len(indices)-len(unused_index)),
#     #       'num unused:', len(unused_index))

#     mixed_x = lam * x + (1 - lam) * x[indices, :]
#     y_a, y_b = y, y[indices]
#     return mixed_x, y_a, y_b, lam


# def smart_train(epoch, train_loader, model, optimizer):
#     train_total = 0
#     train_correct = 0

#     for i, (images, labels, indexes) in enumerate(train_loader):
#         ind = indexes.cpu().numpy().transpose()
#         batch_size = len(ind)

#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()

#         # mixup data
#         inputs, targets_a, targets_b, lam = smart_mixup(images, labels, alpha=8)
#         inputs, targets_a, targets_b = map(
#             Variable, (inputs, targets_a, targets_b))

#         # Forward + Backward + Optimize
#         logits = model(inputs)

#         prec_a, _ = accuracy(logits, targets_a, topk=(1, 5))
#         prec_b, _ = accuracy(logits, targets_b, topk=(1, 5))

#         prec = lam * prec_a + (1-lam)*prec_b
#         # prec = 0.0
#         train_total += 1
#         train_correct += prec

#         # mixup loss
#         loss = lam * F.cross_entropy(logits, targets_a, reduce=True) + (
#             1 - lam) * F.cross_entropy(logits, targets_b, reduce=True)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i+1) % args.print_freq == 0:
#             print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, A Training Accuracy: %.4F, B Training Accuracy: %.4F, Loss: %.4f'
#                   % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, prec_a, prec_b, loss.data))

#     train_acc = float(train_correct)/float(train_total)
#     return train_acc


# def train_avg(epoch, train_loader, model, optimizer):
#     train_total = 0
#     train_correct = 0

#     for i, (images, labels, indexes) in enumerate(train_loader):
#         ind = indexes.cpu().numpy().transpose()
#         batch_size = len(ind)

#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()

#         # mixup data
#         inputs, targets, lam = average_mixup(images, labels)
#         inputs, targets = map(
#             Variable, (inputs, targets))

#         # Forward + Backward + Optimize
#         logits = model(inputs)

#         prec, _ = accuracy(logits, targets, topk=(1, 5))

#         # prec = 0.0
#         train_total += 1
#         train_correct += prec

#         # mixup loss
#         loss = F.cross_entropy(logits, targets, reduce=True)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i+1) % args.print_freq == 0:
#             print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
#                   % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

#     train_acc = float(train_correct)/float(train_total)
#     return train_acc


def train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # mixup data
        inputs, targets_a, targets_b, lam = mixup_data(images, labels)
        inputs, targets_a, targets_b = map(
            Variable, (inputs, targets_a, targets_b))

        # Forward + Backward + Optimize
        print()
        print(targets_a.shape)
        print(targets_b.shape)
        print(inputs.shape)
        logits = model(inputs)

        prec_a, _ = accuracy(logits, targets_a, topk=(1, 5))
        prec_b, _ = accuracy(logits, targets_b, topk=(1, 5))

        prec = lam * prec_a + (1-lam)*prec_b
        # prec = 0.0
        train_total += 1
        train_correct += prec

        # mixup loss
        loss = lam * F.cross_entropy(logits, targets_a, reduce=True) + (
            1 - lam) * F.cross_entropy(logits, targets_b, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, A Training Accuracy: %.4F, B Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, prec_a, prec_b, loss.data))

    train_acc = float(train_correct)/float(train_total)
    return train_acc
# test
# Evaluate the Model


def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    # print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human)

noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels),
      train_dataset.train_labels[:10])
# load model
print('building model...')
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
model.cuda()


epoch = 0
train_acc = 0

# training
file = open("normal_mixup.txt", "a")
max_test = 0

noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    # train_acc = smart_train(epoch, train_loader, model, optimizer)
    train_acc = train(epoch, train_loader, model, optimizer)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    if test_acc > max_test:
        max_test = test_acc
    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    file.write("\nepoch: "+str(epoch))
    file.write("\ttrain acc on train images is "+str(train_acc)+"\n")
    file.write("\ttest acc on test images is "+str(test_acc)+"\n")
file.write("\n\nfinal test acc on test images is "+str(test_acc)+"\n")
file.write("max test acc on test images is "+str(max_test)+"\n")
file.close()

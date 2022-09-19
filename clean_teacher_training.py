# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import sys
from scipy.stats import entropy
from data.SmallClean import RandomClean
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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


def teacher_train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total += 1
        train_correct += prec
        loss = F.cross_entropy(logits, labels, reduce=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    train_acc = float(train_correct)/float(train_total)
    return train_acc

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

print("aggre:", num_training_samples)
# print(train_dataset[:500])

clean_train_dataset, clean_test_dataset, clean_num_classes, clean_num_training_samples = input_dataset(
    args.dataset, 'clean_label', args.noise_path, args.is_human)

small_clean = RandomClean(clean_train_dataset.train_data,
                          clean_train_dataset.train_labels)


print('building model...')
teacher_model = ResNet34(num_classes)
print('building model done')
teacher_optimizer = torch.optim.SGD(
    teacher_model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset=small_clean,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=clean_test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)

alpha_plan = [0.1] * 60 + [0.01] * 40
teacher_model.cuda()

epoch = 0
teacher_train_acc = 0
student_test_acc = 0

print(clean_train_dataset.train_data.shape)

# training
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(teacher_optimizer, epoch, alpha_plan)
    teacher_model.train()

    teacher_train_acc, teacher_train(
        epoch, train_loader, teacher_model, teacher_optimizer)

    # evaluate models
    teacher_test_acc = evaluate(test_loader=test_loader, model=teacher_model)
    # save results
    print('teacher train acc on train images is ', teacher_train_acc)
    print('teacher test acc on test images is ', teacher_test_acc)

teacher_model.save(teacher_model.state_dict(), 'teacher_model.pt')

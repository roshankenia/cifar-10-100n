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


def calculate_entropy(logits):
    # make prediction
    predictions = torch.sigmoid(logits)
    entropies = []

    # calculate entropy for each prediction
    for prediction in predictions:
        entropies.append(entropy(prediction))

    return entropies


def train(epoch, train_loader, teacher_model, teacher_optimizer, student_model, student_optimizer):
    teacher_train_total = 0
    teacher_train_correct = 0

    student_train_total = 0
    student_train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        teacher_logits = teacher_model(images)
        student_logits = student_model(images)

        # obtain entropies
        teacher_entropies = calculate_entropy(teacher_logits.detach().cpu())
        student_entropies = calculate_entropy(student_logits.detach().cpu())

        # sort
        teacher_entropy_indexes = np.argsort(teacher_entropies)
        student_entropy_indexes = np.argsort(student_entropies)
        # calculate number to use
        num_use = int(.5*len(indexes))
        # select samples with lowest entropy
        teacher_entropy_indexes = teacher_entropy_indexes[0:num_use]
        student_entropy_indexes = student_entropy_indexes[0:num_use]

        # find indexes in common
        entropy_in_common = [
            ind for ind in teacher_entropy_indexes if ind in student_entropy_indexes]
        entropy_unlabeled = [
            ind for ind in indexes if ind not in entropy_in_common]
        # only update teacher based on these in common
        print('Teacher being trained on:', len(entropy_in_common))

        teacher_prec, _ = accuracy(teacher_logits, labels, topk=(1, 5))
        # prec = 0.0
        teacher_train_total += 1
        teacher_train_correct += teacher_prec
        teacher_loss = F.cross_entropy(
            teacher_logits[entropy_in_common], labels[entropy_in_common], reduce=True)

        teacher_optimizer.zero_grad()
        teacher_loss.backward()
        teacher_optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Teacher Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, teacher_prec, teacher_loss.data))

        # update student on all with distillation by teacher
        teacher_outputs_unlabeled = teacher_model(images[entropy_unlabeled])

        student_prec, _ = accuracy(teacher_logits, labels, topk=(1, 5))
        # prec = 0.0
        student_train_total += 1
        student_train_correct += teacher_prec
        student_loss = F.cross_entropy(student_logits[entropy_in_common], labels[entropy_in_common], reduce=True) + F.cross_entropy(
            student_logits[entropy_unlabeled], teacher_outputs_unlabeled[entropy_in_common], reduce=True)

        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Student Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, student_prec, student_loss.data))

    teacher_train_acc = float(teacher_train_correct)/float(teacher_train_total)
    student_train_acc = float(student_train_correct)/float(student_train_total)
    return teacher_train_acc, student_train_acc
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

print("aggre:", num_training_samples)
# print(train_dataset[:500])

# clean_train_dataset, clean_test_dataset, clean_num_classes, clean_num_training_samples = input_dataset(
#     args.dataset, 'clean_label', args.noise_path, args.is_human)

# print("clean:", clean_num_training_samples)
# exit()
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels),
      train_dataset.train_labels[:10])
# load model
print('building model...')
teacher_model = ResNet34(num_classes)
student_model = ResNet34(num_classes)
print('building model done')
teacher_optimizer = torch.optim.SGD(
    teacher_model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
student_optimizer = torch.optim.SGD(
    student_model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
student_model.cuda()
teacher_model.cuda()

print("hello")
epoch = 0
teacher_train_acc = 0
student_test_acc = 0

# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(teacher_optimizer, epoch, alpha_plan)
    adjust_learning_rate(student_optimizer, epoch, alpha_plan)
    teacher_model.train()
    student_model.train()

    teacher_train_acc, student_train_acc = train(epoch, train_loader, teacher_model,
                              teacher_optimizer, student_model, student_optimizer)
    # evaluate models
    teacher_test_acc = evaluate(test_loader=test_loader, model=teacher_model)
    student_test_acc = evaluate(test_loader=test_loader, model=student_model)
    # save results
    print('teacher train acc on train images is ', teacher_train_acc)
    print('teacher test acc on test images is ', teacher_test_acc)

    print('\nstudent train acc on train images is ', student_train_acc)
    print('student test acc on test images is ', student_test_acc)
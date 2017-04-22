#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def arguments():
    parser = argparse.ArgumentParser(description='Banjin Vision Trainer')
    parser.add_argument('--t', type=str, metavar='train.pkl',
                        help='Training Data Set File Path')
    parser.add_argument('--v', type=str, metavar='validate.pkl',
                        help='Validation Data Set File Path')
    parser.add_argument('--tbs', type=int, default=64, metavar='N',
                        help='Training Batch Size (64)')
    parser.add_argument('--vbs', type=int, default=1000, metavar='N',
                        help='Validation Batch Size (1024)')
    parser.add_argument('--epochs', type=int, default=128, metavar='N',
                        help='number of epochs (128)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate  0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (1) negative to disable')    
    return parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4_drop = nn.Dropout2d(p=0.25)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(576, 27)

    def forward(self, x):
        x = F.max_pool2d(self.conv2_drop(F.elu(self.conv2(F.elu(self.conv1(x))))), 2)
        x = F.max_pool2d(self.conv4_drop(F.elu(self.conv4(F.elu(self.conv3(x))))), 2)
        x = F.max_pool2d(self.conv6_drop(F.elu(self.conv6(F.elu(self.conv5(x))))), 2)
        x = x.view(-1, 576)
        x = self.fc1(x)
        return F.log_softmax(x)

def run(epoch, args, model, X, y, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    total_loss = 0
    counter = 0
    for i in range(0, y.size(0) - args.tbs + 1, args.tbs):
        data = Variable(X[i : i + args.tbs])
        target = Variable(y[i : i + args.tbs, 0])
        if optimizer:
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.data[0]
        counter += 1
    return total_loss / counter
    
def main():
    # Load arguments
    args = arguments()
    
    # Seed if needed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Load data
    with open(args.t, 'rb') as f:
        X_train, y_train = pickle.load(f)
        X_train = torch.FloatTensor(np.array(X_train, dtype=np.float)).cuda()
        y_train = torch.LongTensor(np.array(y_train[:, :-1], dtype=np.long)).cuda()
    with open(args.v, 'rb') as f:
        X_validate, y_validate = pickle.load(f)
        X_validate = torch.FloatTensor(np.array(X_validate, dtype=np.float)).cuda()
        y_validate = torch.LongTensor(np.array(y_validate[:, :-1], dtype=np.long)).cuda()

    # Prepare model and optimizer
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train
    for epoch in range(0, args.epochs + 1):
        t0 = time.time()
        train_loss = run(epoch, args, model, X_train, y_train,
                         optimizer if epoch > 0 else None)
        t1 = time.time()
        validate_loss = run(epoch, args, model, X_validate, y_validate)
        t2 = time.time()
        print("Epoch %3i Train: %9.6f in %6.3f Validate: %9.6f in %6.3f" %
              (epoch, train_loss, t1 - t0, validate_loss, t2 - t1))

# Run program
if __name__ == "__main__":
    main()

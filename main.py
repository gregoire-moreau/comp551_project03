import torch
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from bounding_boxes import biggest_box
from torch.utils.data import Dataset, Subset

# adapted from https://github.com/floydhub/mnist
# also used https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://pytorch.org/docs/stable/torchvision/transforms.html
# https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
# train_images = pd.read_pickle('input/train_images.pkl')
#train_labels = pd.read_csv('input/train_labels.csv')
# https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/
# https://pytorch.org/docs/stable/optim.html

params = {'dataroot':'/input/',
          'learning_rate':0.2,
          'evalf' : '',
          'outf':'models',
          'ckpf':'',
          'batch_size':64,
          'test_batch_size':1000,
          'epochs':10,
          'momentum':0.25,
          'seed':42,
          'log_interval':10,
          'train':True,
          'train2':True,
          'evaluate': False,
          }

class PickleDataset(Dataset):
    def __init__(self, pkl_file, label_file, transform=None):
        self.images = pd.read_pickle(pkl_file)
        if label_file != '':
            self.labels = pd.read_csv(label_file)
            self.has_labels = True
        else:
            self.has_labels = False

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = transform_image(self.images[idx])
        if self.has_labels:
            target = self.labels.iloc[idx]['Category']
            if self.transform:
                data = self.transform(data)
                return (data, target)
        else:
            if self.transform:
                data = self.transform(data)
            return data



def transform_image(image):
    digit = biggest_box(image)
    return np.reshape(digit , (1,28,28))

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def train(model, device, train_loader, optimizer, epoch, file = None):
    """Training"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % params["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('{{"metric": "Train - NLL Loss", "value": {}}}'.format(
        loss.item()))
            if file:
              file.write(str(epoch)+','+str(100. * batch_idx / len(train_loader))+','+str(loss.item())+'\n')

def test(model, device, test_loader, epoch, file=None):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('{{"metric": "Eval - NLL Loss", "value": {}, "epoch": {}}}'.format(
        test_loss, epoch))
    print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
        100. * correct / len(test_loader.dataset), epoch))
    if file:
      file.write(str(epoch)+","+str(100. * correct / len(test_loader.dataset))+","+str(test_loss)+"\n")



def test_image():
    dataset = PickleDataset('test_images.pkl', '', transform=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]))

    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1, **kwargs)

    # Name generator
    names = iter(list(range(dataset.__len__())))
    model.eval()
    print("evaluating")
    f = open("results2.csv", "w")
    f.write("Id,Category\n")
    with torch.no_grad():
        for data in eval_loader:
            data  = data.to(device, dtype=torch.float)
            output = model(data)
            label = output.argmax(dim=1, keepdim=True).item()
            f.write(str(next(names))+","+str(label)+"\n")
    f.close()


def adapt_lr(initial_lr, epoch):
  return initial_lr * (1/epoch)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print("use cuda ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    try:
        os.makedirs(params["outf"])
    except OSError:
        pass
    torch.manual_seed(params["seed"])
    if use_cuda:
        torch.cuda.manual_seed(params["seed"])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if params["train2"]:
        dataset = PickleDataset('train_images.pkl', 'train_labels.csv', transform=transforms.Compose([
                           # transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        train_loader = torch.utils.data.DataLoader(
            # dataset,
            Subset(dataset, list(range(0, int(0.8*len(dataset))))),
            # Subset(dataset, list(range(0, 128))),
            batch_size=params["batch_size"], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            Subset(dataset, list(range(int(0.8* len(dataset)), len(dataset)))),
            batch_size=params["test_batch_size"], shuffle=True, **kwargs)

    model = Net().to(device)
    losses = open("losses.csv", 'w')
    losses.write("Epoch,Percentage,Loss\n")
    accuracies = open("acc.csv", 'w')
    accuracies.write("Epoch,Accuracy,Loss\n")
    if params["ckpf"] != '':
        if use_cuda:
            model.load_state_dict(torch.load(params["ckpf"]))
        else:
            # Load GPU model on CPU
            model.load_state_dict(torch.load(params["ckpf"], map_location=lambda storage, loc: storage))

    # optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=params["momentum"],nesterov=True)
    # Train?
    if params["train"]:
        # Train + Test per epoch
        for epoch in range(1, params["epochs"] + 1):
            optimizer = optim.SGD(model.parameters(), lr=adapt_lr(params["learning_rate"], epoch), momentum=params["momentum"],
                                  dampening=0, nesterov = True)
            print(optimizer)
            train(model, device, train_loader, optimizer, epoch, file=losses)
            test(model, device, test_loader, epoch, file=accuracies)
            # params["learning_rate"] *= 0.8
        # Do checkpointing - Is saved in outf
        torch.save(model.state_dict(), '%s/mnist_convnet_model_epoch_%d.pth' % (params["outf"], params["epochs"]))
    # Evaluate?
    losses.close()
    accuracies.close()
    if params["evaluate"]:
        test_image()
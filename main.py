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

params = {'dataroot':'/input/',
          'learning_rate':0.01,
          'evalf' : '',
          'outf':'models',
          'ckpf':'',
          'batch_size':64,
          'test_batch_size':1000,
          'epochs':100,
          'momentum':0.5,
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
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = transform_image(self.images[idx])
        target = self.labels.iloc[idx]['Category']

        if self.transform:
            data = self.transform(data)

        return (data, target)

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



def train(model, device, train_loader, optimizer, epoch):
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

def test(model, device, test_loader, epoch):
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



def test_image():
    """Take images from args.evalf, process to be MNIST compliant
    and classify them with MNIST ConvNet model"""
    def get_images_name(folder):
        """Create a generator to list images name at evaluation time"""
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for f in onlyfiles:
            yield f

    def pil_loader(path):
        """Load images from /eval/ subfolder, convert to greyscale and resized it as squared"""
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
                return img.convert('L').resize((sqrWidth, sqrWidth))

    eval_loader = torch.utils.data.DataLoader(ImageFolder(root=params["evalf"], transform=transforms.Compose([
                       transforms.Resize(28),
                       transforms.CenterCrop(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), loader=pil_loader), batch_size=1, **kwargs)

    # Name generator
    names = get_images_name(os.path.join(params["evalf"], "images"))
    model.eval()
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = output.argmax(dim=1, keepdim=True).item()
            print ("Images: " + next(names) + ", Classified as: " + str(label))




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
        dataset = PickleDataset('input/train_images.pkl', 'input/train_labels.csv', transform=transforms.Compose([
                           # transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        train_loader = torch.utils.data.DataLoader(
            Subset(dataset, list(range(0, int(0.8*len(dataset))))),
            batch_size=params["batch_size"], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            Subset(dataset, list(range(int(0.8 * len(dataset)), len(dataset)))),
            batch_size=params["test_batch_size"], shuffle=True, **kwargs)

    model = Net().to(device)

    if params["ckpf"] != '':
        if use_cuda:
            model.load_state_dict(torch.load(params["ckpf"]))
        else:
            # Load GPU model on CPU
            model.load_state_dict(torch.load(params["ckpf"], map_location=lambda storage, loc: storage))

    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=params["momentum"])

    # Train?
    if params["train"]:
        # Train + Test per epoch
        for epoch in range(1, params["epochs"] + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, epoch)
        # Do checkpointing - Is saved in outf
        torch.save(model.state_dict(), '%s/mnist_convnet_model_epoch_%d.pth' % (params["outf"], params["epochs"]))
    # Evaluate?
    if params["evaluate"]:
        test_image()
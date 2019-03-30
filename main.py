import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os
import struct
import numpy as np
from PIL import Image


class my_dataset(Dataset):
    def __init__(self, path, kind, transform=None, target_transform=None):
        if kind == "test":
            kind = 't10k'
        self.image_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        self.label_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        self.transform = transform
        self.target_transform = target_transform

        with open(self.label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            self.labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(self.image_path, 'rb') as imgpath:
            magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(self.labels), 784)

    def __getitem__(self, index):
        img = self.images[index].reshape(28, 28)
        label = self.labels[index]
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(label)

    def __len__(self):
        return len(self.images)


epochs = 3 # 数据集通过网络的次数
batch_size_train = 64 # 批大小
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5 # 动量
log_interval = 10

# data loading
train_loader = torch.utils.data.DataLoader(
    my_dataset('../MNIST/', "train", transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    my_dataset('../MNIST/', "test", transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (examples_data, example_targets) = next(examples)
print(examples_data.shape)

# build the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

network = Net()
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)
optimizer.zero_grad()
# keep track of the progress, y -> loss, x -> counter
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # manually set the gradients to zero
        # ∵ PyTorch by default accumulates gradients
        optimizer.zero_grad()
        # produce the output of the network
        output = network(data)
        # compute a negative log-likelihood loss
        loss = F.nll_loss(output, target)
        # collect new gradients
        loss.backward()
        # propagate back into each of the network's parameters
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # save the network and the optimizer
            torch.save(network.state_dict(), 'results\model.pth')
            torch.save(optimizer.state_dict(), 'results\optimizer.pth')


def test():
    network.eval() #?
    test_loss = 0
    correct = 0
    with torch.no_grad(): # X track
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))


test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()

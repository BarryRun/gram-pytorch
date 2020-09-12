import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net.to(device)
print(net)

input = torch.randn(1, 1, 32, 32).to(device)
output = net(input)
print(output)

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1).to(device)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss.grad_fn)

# net.zero_grad()     # zeroes the gradient buffers of all parameters
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
optimizer.step()    # Does the update

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim


class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x


def loss(x, y, mse_loss):
    loss1 = torch.tensor(0.).to('cuda')
    loss2 = torch.tensor(0.).to('cuda')
    loss3 = torch.tensor(0.).to('cuda')
    loss4 = torch.tensor(0.).to('cuda')
    loss5 = torch.tensor(0.).to('cuda')
    for i in range(x.shape[0]):
        dydx = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
        )[0]
        d2ydx = torch.autograd.grad(
            dydx[:, 0], x, grad_outputs=torch.ones_like(dydx[:, 0]), create_graph=True, only_inputs=True
        )[0][:, 0]
        d2ydy = torch.autograd.grad(
            dydx[:, 1], x, grad_outputs=torch.ones_like(dydx[:, 1]), create_graph=True, only_inputs=True
        )[0][:, 1]
        flag = torch.zeros_like(d2ydy).to('cuda')
        loss5 = mse_loss(d2ydx + d2ydy, flag)
        if x[i, 0] == 0:
            loss1 = mse_loss(y[i], torch.tensor(0.).to('cuda'))
        if x[i, 0] == 1:
            loss2 = mse_loss(y[i], torch.tensor(0.).to('cuda'))
        if x[i, 1] == 0:
            loss3 = mse_loss(y[i], torch.tensor(0.).to('cuda'))
        if x[i, 1] == 1:
            if x[i, 0] != 0:
                if x[i, 0] != 1:
                    loss4 = mse_loss(y[i], torch.tensor(1.).to('cuda'))
    return loss1, loss2, loss3, loss4, loss5


class MyData(Dataset):
    def __init__(self, num_inter, num_boundary):
        """
        归一化方式是否正确
        """
        x1 = torch.randint(low=1, high=100, size=(num_inter, 2))
        x2 = torch.randint(low=0, high=101, size=(num_boundary*4, 1)).reshape(-1, 1)
        x3 = torch.zeros((num_boundary, 1))
        x4 = torch.ones((num_boundary, 1))*100
        x_boundary1 = torch.split(x2, dim=0, split_size_or_sections=200)
        b1 = torch.cat((x3, x_boundary1[0]), dim=1)
        b2 = torch.cat((x_boundary1[1], x3), dim=1)
        b3 = torch.cat((x_boundary1[2], x4), dim=1)
        b4 = torch.cat((x4, x_boundary1[3]), dim=1)
        self.x = torch.cat((x1, b1, b2, b3, b4), dim=0)/100
        self.x = torch.FloatTensor(self.x)

    def __getitem__(self, item):
        return self.x[item]

    def __len__(self):
        return self.x.size(0)


def train(model, dataloader, arg):
    Loss = torch.zeros([arg.epoch, 1])
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    MSE_loss = nn.MSELoss('mean')
    for epoch in range(arg.epoch):
        print('This is the {} epoch'.format(epoch))
        loss_flag = 0
        for index, x in enumerate(dataloader):
            x = x.to(arg.device)
            x = torch.tensor(x, requires_grad=True)
            y = model(x)
            loss1, loss2, loss3, loss4, loss5 = loss(x, y, MSE_loss)
            loss_sum = loss1 + loss2 + loss3 + loss4 + loss5

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            loss_flag += loss_sum

        Loss[epoch] = loss_flag
        print('This epoch loss is {}'.format(Loss[epoch]))
    return Loss


if __name__ == '__main__':
    def initial_arg():
        parser = argparse.ArgumentParser()
        parser.add_argument('--epoch', type=int, default=5000, help='Iteration number')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of k model optimizer')
        parser.add_argument('--device', type=str, default='cuda', help='The run device')
        args = parser.parse_args()
        return args
    arg = initial_arg()

    datasets = MyData(num_inter=100, num_boundary=200)
    dataloader = DataLoader(datasets, batch_size=arg.batch_size, shuffle=True)
    model = Feedforward().to('cuda')
    # Loss = train(model, dataloader, arg)
    # x = np.linspace(0, arg.epoch, arg.epoch)
    # Loss = torch.tensor(Loss, requires_grad=False).to('cpu')
    # Loss = Loss.numpy()
    # plt.plot(x, Loss)
    # plt.show()

    # torch.save(model.state_dict(), 'model_' + str(arg.epoch) + '.pkl')
    model.load_state_dict(torch.load('model_' + str(arg.epoch) + '.pkl'))
    with torch.no_grad():
        y = torch.zeros((100, 100)).to('cuda')
        for i in range(100):
            for j in range(100):
                x = torch.FloatTensor([i/100, j/100]).to('cuda')
                x = x.reshape(1, 2)
                y[i, j] = model(x)
                # print(x)
    y = torch.tensor(y, requires_grad=False).to('cpu')
    y = y.numpy()
    y = np.rot90(y, 1) + 0.2
    plt.imshow(y, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.show()





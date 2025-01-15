import torch


class MLP(torch.nn.Module):
    def __init__(self, activation='softmax'):
        super(MLP, self).__init__()
        assert activation in ['relu', 'tanh', 'sigmoid'], 'Invalid activation function'
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 10)
        # self.fc4 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmod = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = self.relu
        if activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'sigmoid':
            self.activation = self.sigmod

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.activation(x)
        # x = self.fc4(x)
        x = self.softmax(x)
        return x

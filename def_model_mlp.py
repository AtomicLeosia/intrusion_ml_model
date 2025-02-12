import torch.nn as nn

class ModelMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ModelMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelDeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ModelDeepMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x):
        x = self.leaky_relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.leaky_relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x
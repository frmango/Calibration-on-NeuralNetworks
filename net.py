import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def init_weights(m):
    """
    Args:
        m (tensor): Tensor whose elemets are replaced with the value returned by callable.
    ----------------------------

    All weights cannot be initialized to the value 0.0,
    as the optimization algorithm results in some asymmetry in the error gradient,
    the He initialization method is calculated as a random number with a Gaussian probability distribution G 
    with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))

class Net(nn.Module):

    def __init__(self, droprate=0.5):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=2)

        # He initialization for linear layers
        for m in self.modules():
             m.apply(init_weights)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(p=droprate)   # adding dropout layer
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)   # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout(x)    # applying dropout to the output of fc1
        logits = self.fc2(x)
        return logits


class LeNet5(nn.Module):

    def __init__(self, n_classes, droprate=0.5):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        # He initialization for convolutional layers
        for m in self.feature_extractor.modules():
            m.apply(init_weights)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Dropout(p=droprate),     # adding dropout layer
            nn.Linear(in_features=84, out_features=10),
        )
        # He initialization for linear layers
        for m in self.classifier.modules():
            m.apply(init_weights)

    def forward(self, x, mc_samples=10):
        predictions = []
        for _ in range(mc_samples):
             features = self.feature_extractor(x)
             features = torch.flatten(features, 1)
             logits = self.classifier(features)
             predictions.append(logits.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        mean_logits = torch.mean(predictions, dim=0)    # average logits
        probs = F.softmax(mean_logits, dim=1)    # softmax probabilities based on the averaged logits
        return mean_logits,probs
    


class VGG16(nn.Module):

    def __init__(self, num_classes=10):
        # calling constructor of parent class
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer1.apply(init_weights)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2.apply(init_weights)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3.apply(init_weights)

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer4.apply(init_weights)

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5.apply(init_weights)

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6.apply(init_weights)

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer7.apply(init_weights)

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer8.apply(init_weights)

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9.apply(init_weights)

        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer10.apply(init_weights)

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer11.apply(init_weights)

        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12.apply(init_weights)

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) # Max pooling over a (2, 2) window
        self.layer13.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
     
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1) # flatten all dimensions except the batch dimension
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
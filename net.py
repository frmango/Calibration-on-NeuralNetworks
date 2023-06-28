import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

from layers.BBB.BBBConv import BBB_Conv2d 
from layers.BBB.BBBLinear import BBB_Linear

from layers.BBB_LRT.BBBConv import BBB_LRT_Conv2d
from layers.BBB_LRT.BBBLinear import BBB_LRT_Linear

from layers.misc import FlattenLayer, ModuleWrapper

def init_weights_h(m):
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

def init_weights(m):
    """
    Args:
        m (tensor): Tensor whose elemets are replaced with the value returned by callable.
    ----------------------------

    All weights cannot be initialized to the value 0.0,
    as the optimization algorithm results in some asymmetry in the error gradient,
    the Xavier initialization method is employed
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(1. / n))


class Net(nn.Module):

    def __init__(self, droprate=0.5):
        '''A basic architecture'''

        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=2)

        # He initialization for linear layers
        for m in self.modules():
             m.apply(init_weights_h)    # He weight initialization works with Rectified Linear Unit (ReLU) activation function 

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
    '''The architecture of LeNet'''

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
        # Xavier initialization for linear layers
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
    

class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinear(5 * 5 * 16, 120, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(84, outputs, bias=True, priors=self.priors)


    # No need to define forward method. It'll automatically be taken care of by ModuleWrapper


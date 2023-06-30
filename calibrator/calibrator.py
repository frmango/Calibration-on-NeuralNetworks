from typing import Sequence
import numpy as np
import scipy
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F

from calibrator.attention_ts import CalibAttentionLayer
from gnet import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intra_distance_loss(output, labels):
    """
    Marginal regularization from CaGCN (https://github.com/BUPT-GAMMA/CaGCN)
    """
    output = F.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = pred_max_index==labels
    incorrect_i = pred_max_index!=labels
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    incorrect_loss = torch.sum(pred[incorrect_i]-sub_pred[incorrect_i]) / labels.size(0)
    correct_loss = torch.sum(1- pred[correct_i] + sub_pred[correct_i]) / labels.size(0)
    return incorrect_loss + correct_loss

def fit_calibration(temp_model, eval, data, train_mask, test_mask, patience = 100):
    """
    Train calibrator
    """    
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(data.x, data.edge_index)
        labels = data.y
        edge_index = data.edge_index
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


# temperature scaling
class TS(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self

class CaGCN(nn.Module):
    def __init__(self, model, num_nodes, num_class, dropout_rate):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagcn = GCN(num_class, 1, 16, drop_rate=dropout_rate, num_layers=2)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits, edge_index)
        return logits * F.softplus(temperature)

    def graph_temperature_scale(self, logits, edge_index):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, edge_index)
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, data.edge_index)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


class GATS(nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_class, dist_to_train, gats_args):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=gats_args.heads,
                                         bias=gats_args.bias)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


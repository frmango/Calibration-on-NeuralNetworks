""" 
Temperature scaling calibration technique
=====================

In the multiclass context, the network outputs a class prediction y and confidence score p for each input. 
The network logits z are vectors, where p_hat is typically derived using the softmax function $\sigma _{SM}$.

----------------------------

Temperature scaling is the simplest extension of Platt scaling, since it uses a single scalar parameter T > 0 for all classes.
Since the parameter T does not change the maximum of the softmax function, the class prediction  remains unchanged.
Hence it does not affect the model's accuracy.

T is optimized w.r.t. NLL on the validation set. 

"""
import torch
from torch import nn
from torch.optim import LBFGS
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

def accuracy(predictions, labels):
    pred_labels = torch.argmax(predictions, dim=1)
    correct_count = torch.sum(pred_labels == labels)
    return correct_count.float() / labels.size(0)


def temp_scaling(logits_nps, labels_nps, maxiter=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temperature = torch.tensor([1.5], device=device, requires_grad=True)   # initialization value doesn't seem to matter that much based on some ad-hoc experimentation
    
    logits_tensor = torch.tensor(logits_nps, device=device)
    labels_tensor = torch.tensor(labels_nps, device=device)
    
    # Compute INITIAL accuracy on validation set
    with torch.no_grad():
        valid_preds = F.softmax(logits_tensor, dim=1)   # returns softmax-ed confidence scores
        acc = accuracy(valid_preds, labels_tensor).item() * 100
        # print(f'Validation accuracy: {acc:.3f}%')
    
    
    logits_w_temp = logits_tensor / temperature    # logits, to be used with cross entropy loss
    nll_loss = F.cross_entropy(logits_w_temp, labels_tensor)
    org_nll_loss = nll_loss.clone().detach()    # copy of the tensor, gradients are not calculated for it during backpropagation

    
    #def closure():
     #   optimizer.zero_grad()
     #   loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)   # Compute negative log-likelihood loss
     #   loss.backward()
     #   return loss


    # Use L-BFGS optimizer to find optimal temperature
    # optimizer = torch.optim.LBFGS([temperature], max_iter=maxiter)
    optimizer = torch.optim.SGD([temperature], lr=0.0001)
    #optimizer.step(closure)
    optimizer.step()
    
    # Compute accuracy and NLL loss AFTER temperature scaling
    with torch.no_grad():
        valid_preds_scaled = F.softmax(logits_w_temp, dim=1)
        acc_scaled = accuracy(valid_preds_scaled, labels_tensor).item() * 100
        nll_loss_scaled = F.cross_entropy(logits_tensor / temperature, labels_tensor).item()
        
    print(f'Original NLL: {org_nll_loss:.3f}, validation accuracy: {acc:.3f}%')
    print(f'After temperature scaling, NLL: {nll_loss_scaled:.3f}, validation accuracy: {acc_scaled:.3f}%')
    print(f'Temperature: {temperature.item():.3f}')
    
    return temperature

def calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def make_model_diagrams(outputs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct.cpu().numpy()  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = calculate_ece(outputs, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('reliability_diagram.png')
    plt.show()
    return ece
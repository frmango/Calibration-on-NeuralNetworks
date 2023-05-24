import os
import argparse
import numpy as np
import torch

from net import * 
from config import parse_args
from CIFAR10 import train_loader,test_loader, val_loader
from temperature_scaling import temp_scaling, calculate_ece, make_model_diagrams
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, criterion, optimizer, device):
    """train model for one epoch
    Args:
        model (torch.nn.Module): model to train
        train_loader (object): iterator to load data
        criterion (torch.nn.Module): loss function
        optimizer (torch.nn.optim): stochastic optimzation strategy
        device (str): device to train model ('cpu' or 'cuda')
        epoch (int): current epoch
    """
    model.train()
    running_loss = 0.0
    step = 0
    for inputs, labels in train_loader:
        # forward pass
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, probs = model(inputs)
        # back propogation
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # log
        step = step + 1
        log_interval = 1000
        if step % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (step+1) * len(inputs), len(train_loader.dataset),
                    100. * (step+1) / len(train_loader), loss.item()
                )
            )

        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    """Evaluate model performance with validation data
    Args:
        model (torch.nn.Module): model to evaluate
        val_loader (object): iterator to load data
        criterion (torch.nn.Module): loss function
        device (str): device to evaluate model ('cpu' or 'cuda')
    """
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #print('Input shape:', inputs.shape)
            #print('Labels shape:', labels.shape)

            logits, probs = model(inputs)
            loss = criterion(logits, labels)
            test_loss += loss.item() * inputs.size(0)  # sum up batch loss
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = test_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print(
        '\nValidation set - average loss: {:.4f}, '.format(epoch_loss)
    )

    return epoch_loss, epoch_acc


def initialize_model(args):
    """Initialize model checkpoint dictionary for storing training progress
    Args:
        args (object):
            epoch (int): total number of epochs to train model
            n_classes (int): number of segmentation classes
    """
    model_dict = {
        'total_epoch': args.epochs,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': list(),
        'test_loss': list(),
    }
    return model_dict

def get_model(args, device):
    """Intialize or load model checkpoint and intialize model and optimizer
    Args:
        args (object):
            model (str): filename of model to load
                (initialize new model if none is given)
        device (str): device to train and evaluate model ('cpu' or 'cuda')
    """
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model(args)
    
    model = LeNet5(n_classes=10).cuda() if device == 'cuda' else LeNet5(n_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = 0.005, nesterov = True, momentum = args.momentum)

    if args.model:
        model.load_state_dict(model_dict['model_state_dict'])
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model, optimizer, model_dict

if __name__ == '__main__':
    args = parse_args()
    if args.tensorboard:
        writer = SummaryWriter()
    # initialize model
    device = (
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )

    model, optimizer, model_dict = get_model(args, device)
    # define loss function
    criterion = nn.CrossEntropyLoss()

    print ("Data loaded! Building model...")

    # train and evaluate model
    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    n_epoch = start_epoch + args.epochs - 1

    model_path = os.getcwd() + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    model_name = f'models/Lenet5{n_epoch}.pt'

    # Initialize lists to store the test errors
    #test_errors = []
    for epoch in range(start_epoch, n_epoch + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        epoch_loss, epoch_acc = validate(model, val_loader, criterion, device)
        # update tensorboard
        if args.tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', epoch_loss, epoch)
        
        # record training progress
        model_dict['train_loss'].append(train_loss)
        model_dict['test_loss'].append(epoch_loss)

        if args.save:
            torch.save(model_dict, model_name)
            
    if args.tensorboard:
        writer.close()

    # print model statistics
    # print('training loss:', train_losses)
    # print('validation loss:', epoch_losses)

    print ("Model built! Getting logits...")
    # Get logits for all validation or test images
    logits_nps = []
    labels_nps = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits, probs = model(images)
            logits_nps.append(logits)
            labels_nps.extend(labels)

    # Concatenate logits
    logits_nps = np.concatenate(logits_nps)

    print("Logits get! Do temperature scaling...")
    print("=" * 80)

    # Perform temperature scaling on logits_nps
    temp_var = temp_scaling(logits_nps, labels_nps)
    # use temp_var with your logits to get calibrated output

    print("=" * 80)
    print("Done!")

    logits_ts = torch.from_numpy(np.array(logits_nps))
    labels_ts = torch.from_numpy(np.array(labels_nps))

    # ece = make_model_diagrams(logits_ts, labels_ts)
  
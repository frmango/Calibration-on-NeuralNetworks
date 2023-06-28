import argparse

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--batch-size', type=int, default=2, metavar='N',
        help='input batch size for training (default: 2)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=2, metavar='N',
        help='input batch size for testing (default: 2)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, metavar='LR',
        help='learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, metavar='M',
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='number of workers to load data'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='keep batch norm layers with 32-bit precision'
    )
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save', action='store_true', default=False,
        help='save the current model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False,
        help='record training log to Tensorboard'
    )
    
    args = parser.parse_args()
    return args
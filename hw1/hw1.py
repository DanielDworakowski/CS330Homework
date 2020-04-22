import numpy as np
import torch
import random
from load_data import DataGenerator
import argparse
import debug as db
from tqdm.auto import tqdm
import torch.nn.functional as F
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--meta_batch_size', default=16, type=int, help='Number of N-way classification tasks per batch')
    parser.add_argument('--num_classes', default=5, type=int, help='number of classes used in classification (e.g. 5-way classification).')
    parser.add_argument('--num_samples', default=1, type=int, help='number of examples used for inner gradient update (K for K-shot learning).')
    args = parser.parse_args()
    return args

def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N * N] network output
        labels: [B, K+1, N, 1] labels
    Returns:
        scalar loss
    """
    preds = preds[:, -1, :]
    labels = labels[:, -1]
    N = int(preds.shape[1] ** 0.5)
    preds = preds.view(preds.shape[0] * N, -1)
    labels = labels.view(-1)
    return F.cross_entropy(preds, labels)

class MANN(torch.nn.Module):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        # self.layer1 = torch.nn.LSTM(128, return_sequences=True)
        # self.layer2 = torch.nn.LSTM(num_classes, return_sequences=True)
        self.lstm1 = torch.nn.LSTM(self.num_classes * (784 + self.num_classes), 128, batch_first=True)
        self.lstm2 = torch.nn.LSTM(128, num_classes * num_classes, batch_first=True)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #
        # Remove information from final sequence input.
        inlbl = input_labels.clone()
        inlbl[..., -1, :, :].zero_()
        #
        # Create LSTM inputs.
        lstm_input = torch.cat([input_images, input_labels], -1).view(input_images.size(0), input_labels.size(1), -1)
        out1, h1, = self.lstm1(lstm_input)
        out, h2 = self.lstm2(out1)
        return out
#
# Get the device to be used based on whether the GPU is being used.
def getDevice(usegpu, rank):
    return torch.device('cuda:%d'%(rank) if usegpu else 'cpu')
#
# Get a batch sample.
def sampleBatch(generator, gentype, num, device):
    i, l_onehot, l_lbl = generator.sample_batch(gentype, num)
    i = torch.from_numpy(i).to(device)
    l_onehot = torch.from_numpy(l_onehot).to(device)
    l_lbl = torch.from_numpy(l_lbl).to(device)
    return i, l_onehot, l_lbl
#
#
if __name__ == '__main__':
    #
    # Setup optimization.
    device = getDevice(torch.cuda.is_available(), 0)
    args = getInputArgs()
    data_generator = DataGenerator(args.num_classes, args.num_samples + 1)
    mann = MANN(args.num_classes, args.num_samples + 1).to(device)
    optim = torch.optim.Adam(mann.parameters(), 0.001)
    #
    # Train.
    pbar = tqdm(range(50000))
    for step in pbar:
        i, l_onehot, l_lbl = sampleBatch(data_generator, 'train', args.meta_batch_size, device)
        out = mann(i, l_onehot)
        loss = loss_function(out, l_lbl)
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss='{:.2e}'.format(loss.item()))
            # acc='{:.2e}'.format(acc))
        if step % 1 == 0:
            with torch.no_grad():
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                i, l_onehot, l_lbl = sampleBatch(data_generator, 'test', 100, device)
                #
                # Compute loss.
                out = mann(i, l_onehot)
                tls = loss_function(out, l_lbl)
                #
                # Compute Accuracy.
                out = out.view(i.shape[0], i.shape[1], args.num_classes, args.num_classes)
                out = out[:, -1, :, :]
                l_lbl = l_lbl[:, -1, :]
                preds = out.max(-1)

                l_lbl = l_lbl.view(preds[0].shape)
                acc = (l_lbl == preds[1]).float().mean()

                print("Train Loss:", loss, "Test Loss:", tls)
                print("Test Accuracy", acc)

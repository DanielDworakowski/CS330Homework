import numpy as np
import torch
import random
from load_data import DataGenerator
import argparse
import debug as db
from tqdm.auto import tqdm
from torch import nn
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

def loss_function(lsfn, preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, 1] labels
    Returns:
        scalar loss
    """
    # preds = preds[:, -N:]
    # preds = preds.reshape(-1, N)
    preds = MANN.getMetaPreds(preds)
    labels = labels.reshape(-1)
    return lsfn(preds, labels)


class MANNNTM_STATE(torch.nn.Module):

    def __init__(self, bs, hsize, nreads, memory_shape, device):
        self.M = torch.zeros((bs,) + memory_shape, device=device)
        self.read_weight = torch.zeros((bs, nreads, memory_shape[0]), device=device)
        self.used_weight = torch.zeros((bs, memory_shape[0]), device=device)
        self.c = torch.zeros((bs, hsize), device=device)
        self.h = torch.zeros((bs, hsize), device=device)
        self.r = torch.zeros((bs, nreads * memory_shape[1]), device=device)

class MANNNTM(torch.nn.Module):


    def __init__(self, nclass, nreads, input_size, cell_size, memory_shape, gamma):
        """
        Adapted from:
        https://github.com/ywatanabex/ntm-meta-learning/blob/master/utils/models.py

        Args
            nclass (int): number of classes in a episode
            nreads (int): number of read heads
            input_size (int): dimention of input vector
            cell_size (int): cell size of LSTM controller
            memory_shape (tuple of int): num_memory x dim_memory
            gamma (float) : decay parameter of memory
        """
        super().__init__()
        self.nclass = nclass
        self.nreads = nreads
        self.input_size = input_size
        self.cell_size = cell_size
        self.memory_shape = memory_shape
        self.gamma = gamma

        self.l_key = nn.Linear(self.cell_size, self.nreads * self.memory_shape[1])
        self.l_add = nn.Linear(self.cell_size, self.nreads * self.memory_shape[1])
        self.l_sigma = nn.Linear(self.cell_size, 1)
        self.l_ho = nn.Linear(self.cell_size, self.nclass)
        self.l_ro = nn.Linear(self.nreads * self.memory_shape[1], self.nclass)
        self.lstm_rh = nn.Linear(self.nreads * self.memory_shape[1], self.cell_size)
        self.lstm_cell = nn.LSTMCell(input_size, cell_size)


    def cosine_similarity(self, x, y, eps=1e-6):
        n1, n2, n3 = x.shape
        _, m2, _ = y.shape
        z = x @ y.transpose(1, 2)
        x2 = x.pow(2).sum(dim=2).view(n1, n2, 1).expand(-1, -1, m2)
        y2 = y.pow(2).sum(dim=2).view(n1, 1, m2).expand(-1, n2, -1)
        z /= torch.exp((x2 * y2 + eps).log() / 2)
        return z

    def forward_single(self, x_t, state):
        bs = x_t.shape[0]

        state.h = state.h + self.lstm_rh(state.r)
        h_t, c_t = self.lstm_cell(x_t, (state.h, state.c))

        key = self.l_key(h_t).view(bs, self.nreads, self.memory_shape[1])
        add = self.l_add(h_t).view(bs, self.nreads, self.memory_shape[1])
        sigma = self.l_sigma(h_t)
        #
        # Non-differentiable ops.
        with torch.no_grad():
            lu_idx = state.used_weight.argsort(dim=1)[:, :self.nreads]
            wlu_tml_data = torch.zeros((bs, self.memory_shape[0]), device=x_t.device)
            wlu_tml_data.scatter_(1, lu_idx, 1)
        #
        # Back to differentiable.
        # Compute write weights.
        wlu_tml = wlu_tml_data
        _wlu_tml = wlu_tml.view(bs, 1, self.memory_shape[0]).expand(-1, self.nreads, -1)
        _sigma = sigma.view(bs, 1, 1).expand(-1, self.nreads, self.memory_shape[0])
        ww_t = _sigma * state.read_weight + (1 - _sigma) * _wlu_tml
        #
        # Write to memory.
        _lu_mask = wlu_tml.view(bs, self.memory_shape[0], 1).expand(-1, -1, self.memory_shape[1])
        M_t = state.M * _lu_mask + ww_t.transpose(1, 2) @ add
        #
        # Read from memory.
        K_t = self.cosine_similarity(key, M_t)
        # K_t = F.cosine_similarity(key, M_t, 1)
        wr_t = F.softmax(K_t.view(-1, self.memory_shape[0]), dim=-1).view(bs, self.nreads, self.memory_shape[0])
        wu_t = self.gamma * state.used_weight + wr_t.sum(dim=1) + ww_t.sum(dim=1)

        r_t = (wr_t @ M_t).view(bs, -1)
        state.M = M_t
        state.read_weight = wr_t
        state.used_weight = wu_t
        state.c = c_t
        state.h = h_t
        state.r = r_t
        return self.l_ho(h_t) + self.l_ro(r_t), state

    def forward(self, x):
        state = MANNNTM_STATE(x.shape[0], self.cell_size, self.nreads, self.memory_shape, x.device)
        out = []
        for t in range(x.shape[1]):
            x_t = x[:, t]
            out_t, state = self.forward_single(x_t, state)
            out.append(out_t)
        out = torch.stack(out, 1)
        return out

class MANN(torch.nn.Module):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        hidden = 256
        # self.lstm1 = torch.nn.LSTM((784 + self.num_classes), hidden)
        # (self, nclass, nreads, input_size, cell_size, memory_shape, gamma):
        self.lstm1 = MANNNTM(nclass=self.num_classes, nreads=4, input_size=28*28 + self.num_classes, cell_size=hidden, memory_shape=(128, 40), gamma=0.95,)
        #model = NTM(nb_class=5, nb_reads=4, input_size=28*28, cell_size=200, memory_shape=(128, 40), gamma=0.95
        #  self.lstm1 = MANNNTM(nclass=5, nreads=4, input_size=28*28 + self.num_classes, cell_size=hidden, memory_shape=(128, 40), gamma=0.95,)
        self.lstm2 = torch.nn.LSTM(hidden, num_classes)

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
        input_labels = input_labels.clone()
        input_labels[..., -1, :, :].zero_()
        #
        # Create LSTM inputs, transpose since LSTM takes features first.
        lstm_input = torch.cat([input_images, input_labels.view(input_images.shape[0], input_images.shape[1], -1)], -1)
        lstm_input = lstm_input.view(input_images.size(0), input_labels.size(1) * self.num_classes, -1)
        db.printTensor(lstm_input)
        out = self.lstm1(lstm_input)
        # out, h2 = self.lstm2(out1)
        # out = out.transpose(0, 1)
        return out

    @classmethod
    def getMetaPreds(cls, preds):
        N = preds.shape[-1]
        preds = preds[:, -N:]
        preds = preds.reshape(-1, N)
        return preds

# Get the device to be used based on whether the GPU is being used.
def getDevice(usegpu, rank):
    usegpu=False
    return torch.device('cuda:%d'%(rank) if usegpu else 'cpu')
# #
# # Get a batch sample.
# def sampleBatch(generator, gentype, num, device, args):
#     i, l_onehot = generator.sample_batch(gentype, num)
#     i = torch.from_numpy(i).to(device)
#     l_onehot = torch.from_numpy(l_onehot).to(device)
#     last_n_step_labels = l_onehot[:, -1:]
#     last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, args.num_classes)  # (B * N, N)
#     l_lbl = torch.tensor(last_n_step_labels.argmax(axis=1)).to(device)
#     return i.float(), l_onehot.float(), l_lbl.long()

#
# Get a batch sample.
def sampleBatch(generator, gentype, num, device, args):
    i, l_onehot, l_lbl = generator.sample_batch(gentype, num)
    i = torch.from_numpy(i).to(device)
    l_onehot = torch.from_numpy(l_onehot).to(device)
    l_lbl = torch.from_numpy(l_lbl)[:, -1:].view(-1).to(device)
    # last_n_step_labels = l_onehot[:, -1:]
    # last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, args.num_classes)  # (B * N, N)
    # l_lbl2 = torch.tensor(last_n_step_labels.argmax(axis=1)).to(device)
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
    ce = torch.nn.CrossEntropyLoss()
    #
    # Train.
    pbar = tqdm(range(50000))
    for step in pbar:
        i, l_onehot, l_lbl = sampleBatch(data_generator, 'train', args.meta_batch_size, device, args)

        out = mann(i, l_onehot)
        loss = loss_function(ce, out, l_lbl)
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss='{:.4e}'.format(loss.item()))
            # acc='{:.2e}'.format(acc))
        if step % 100 == 0:
            with torch.no_grad():
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                i, l_onehot, l_lbl = sampleBatch(data_generator, 'test', 100, device, args)
                #
                # Compute loss.
                out = mann(i, l_onehot)

                tls = loss_function(ce, out, l_lbl)
                #
                # Compute Accuracy.
                out = MANN.getMetaPreds(out)
                _, preds_idx = out.max(-1)
                l_lbl = l_lbl.view(preds_idx.shape)
                acc = (l_lbl == preds_idx).float().mean()

                print("Train Loss:", loss, "Test Loss:", tls)
                print("Test Accuracy", acc)

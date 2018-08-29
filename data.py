import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

class DataSet(Dataset):
    def __init__(self, fname):
        super(DataSet, self).__init__()

        self.data = ''
        self.corpus = []
        with open(fname, 'r') as f:
            self.data = list(f.read().split(' '))
            self.corpus = list(set(list(self.data)))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, i):
        inputvalue = one_hot(self.getLabelFromChar(self.data[i]), len(self.corpus))
        inputvalue = torch.Tensor(inputvalue)
        targetvalue = torch.Tensor([[self.getLabelFromChar(self.data[i+1])]])
        return [ inputvalue, targetvalue ]

    def sequentialSampler(self):
        return DataLoader(self, sampler=SequentialSampler(self), 
                batch_size=10, num_workers=4)

    def getLabelFromChar(self, c):
        return self.corpus.index(c)

    def getCharFromLabel(self, l):
        return self.corpus[l]

def one_hot(l, classes):
    x = torch.zeros(classes)
    x[l] = 1
    return x

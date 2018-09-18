import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler, Sampler
from random import shuffle

class DataSet(Dataset):
    def __init__(self, fname):
        super(DataSet, self).__init__()

        self.data = ''
        self.corpus = []
        with open(fname, 'r') as f:
            self.data = list(f.read())
            self.corpus = list(set(list(self.data)))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, i):
        start, end = self.data[i], self.data[i+1]
        inputvalue = torch.zeros(1, len(self.corpus))
        inputvalue[0] = one_hot(self.getLabelFromChar(start), len(self.corpus))
        targetvalue = torch.Tensor([[self.getLabelFromChar(end)]])
        return [ inputvalue, targetvalue ]

    def sequentialSampler(self, batch_size):
        return DataLoader(self, sampler=RandomSequentialSampler(self, 800), 
                batch_size=batch_size, num_workers=8)

    def getLabelFromChar(self, c):
        return self.corpus.index(c)

    def getCharFromLabel(self, l):
        return self.corpus[l]

def one_hot(l, classes):
    x = torch.zeros(classes)
    x[l] = 1
    return x

class RandomSequentialSampler(Sampler):
    def __init__(self, datasource, seqCount):
        self.source = datasource
        self.seqCount = seqCount

    def __iter__(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        groups = [[range(i, i + self.seqCount)] for i in range(0,len(self),self.seqCount)]
        shuffle(groups)
        groups = flatten(flatten(groups))
        return iter(groups)

    def __len__(self):
        return len(self.source) - self.seqCount

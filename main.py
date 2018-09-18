#!/usr/bin/env python3
import torch
import sys
from model import Model
from data import DataSet, one_hot
from random import choice

batch_size = 150

def output(startChar, length, model, dataset):
    def fakeDataset(character):
        x = torch.zeros(batch_size, 1, len(dataset.corpus))
        x[0][0] = one_hot(dataset.getLabelFromChar(character), len(dataset.corpus))
        y = None
        return [x,y]

    def getChar(x):
        label = list(x[0])
        label = label.index(max(label))
        char = dataset.getCharFromLabel(label)
        return char

    x = fakeDataset(startChar)
    outputString = startChar

    while len(outputString) < length:
        x = model.train(x , adjust=False)
        c = getChar(x)
        outputString += c
        x = fakeDataset(c)
    return outputString

def main():
    dataset = DataSet('Datafiles/tiny-shakespeare.txt')
    corpus = dataset.corpus
    model = Model(len(corpus), batch_size)
    
    i = 0

    while True:
        for j, datapoint in enumerate(dataset.sequentialSampler(batch_size)):
            if datapoint[0].size(0) == batch_size:
                model.train(datapoint)
                if i % 1500 == 0 and i > 0:
                    print(output(choice(corpus), 1000, model, dataset), 
                            file=sys.stderr)
                i += 1
main()

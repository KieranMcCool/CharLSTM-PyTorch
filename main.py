#!/usr/bin/env python3
import torch
import sys
from model import Model
from data import DataSet
from random import choice

def output(startChar, length, model, dataset):
    def fakeDataset(character):
        x = torch.Tensor([[dataset.getLabelFromChar(character)]])
        y = None
        return [x,y]

    def getChar(x):
        label = list(x)
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
    dataset = DataSet('index.html')
    corpus = dataset.corpus
    model = Model(1, len(corpus))
    
    i = 0
    while True:
        for j, datapoint in enumerate(dataset.sequentialSampler()):
            if datapoint[0].size(0) == 10:
                model.train(datapoint)
                """
                if i % 1000 == 0 and i > 0:
                    print(output(choice(corpus), 100, model, dataset), 
                            file=sys.stderr)
                """
                i += 1
        print(output(choice(corpus), 100, model, dataset), 
                file=sys.stderr)
main()

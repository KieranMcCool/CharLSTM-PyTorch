import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

learning_rate = 1e-3
CUDA = False

class Model(nn.Module):

    def __init__(self, vectorSize, numClasses):
        super(Model, self).__init__()

        self.iteration = 0
        self.vectorSize = numClasses 
        self.hiddenSize = 64
        self.numLayers = 3

        self.crossEntropy = True
        if self.crossEntropy:
            self.numClasses = numClasses
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.numClasses = 1
            self.loss_fn = nn.MSELoss()

        self.lstm = nn.LSTM(self.vectorSize, self.hiddenSize, self.numLayers, dropout=0.1)
        self.fc = nn.Sequential(
                #nn.Softmax(dim=1),
                nn.Linear(self.vectorSize * self.hiddenSize, self.numClasses))
        self.hidden = self.init_hidden()
        self.old_hidden = self.hidden
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if torch.cuda.is_available() and CUDA:
            self.cuda()

    def init_hidden(self):
        return (Variable(torch.zeros(self.numLayers, self.vectorSize, self.hiddenSize)),
        Variable(torch.zeros(self.numLayers, self.vectorSize, self.hiddenSize)))

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train(self, data, adjust=True):

        inputVector = data[0]
        targetVector = data[1]

        if torch.cuda.is_available() and CUDA:
            inputVector = inputVector.cuda()
            targetVector = targetVector.cuda()

        x = Variable(inputVector)
        y = Variable(targetVector, requires_grad=False).squeeze()

        y_prediction = None

        if adjust:
            self.optimizer.zero_grad()
            #self.hidden = self.init_hidden()
            y_prediction = self(x)
            loss = self.loss_fn(y_prediction, y.long() if self.crossEntropy else y)
            if self.iteration == 1:
                loss.backward(retain_graph=True)
            self.optimizer.step()
            self.iteration += 1

            if self.iteration % 10 == 0 and True:
                print(torch.mean(loss.data))

        else:
            y_prediction = self(x).squeeze()

        return y_prediction.data

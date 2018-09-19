import torch
import torch.nn as nn
import torch.nn.functional as f
import gc
from torch.autograd import Variable

learning_rate = 1e-3
torch.backends.cudnn.enabled = False
CUDA = True

class Model(nn.Module):

    def __init__(self, numClasses, batch_size):
        super(Model, self).__init__()

        self.iteration = 0
        self.batch_size = batch_size
        self.vectorSize = numClasses
        self.hiddenSize = 512
        self.numLayers = 4

        self.crossEntropy = True
        if self.crossEntropy:
            self.numClasses = numClasses
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.numClasses = 1
            self.loss_fn = nn.MSELoss()

        self.lstm = nn.LSTM(self.vectorSize, self.hiddenSize, self.numLayers, 
                dropout=0.1, batch_first=True)
        self.fc = nn.Sequential(
               nn.BatchNorm1d(self.hiddenSize),
               nn.Softmax(dim=1),
                nn.Linear(self.hiddenSize, self.numClasses))
        self.hidden = self.init_hidden()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if torch.cuda.is_available() and CUDA:
            self = self.cuda()

    def init_hidden(self):
        def gen_tensor():
            return torch.zeros(self.numLayers, self.batch_size, self.hiddenSize)
        x = gen_tensor()
        y = gen_tensor()

        if torch.cuda.is_available() and CUDA and True:
            x = x.cuda()
            y = y.cuda()
        return (Variable(x),Variable(y))

    def repackage_hidden(self): 
        x = self.hidden[0].data
        y = self.hidden[1].data
        self.hidden = (Variable(x), Variable(y))

        
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

        x = Variable(inputVector)
        y_prediction = self(x)

        self.repackage_hidden()

        if adjust:

            if torch.cuda.is_available() and CUDA:
                targetVector = targetVector.cuda()

            y = Variable(targetVector, requires_grad=False).squeeze()
            self.optimizer.zero_grad()
            y_prediction = self(x)
            loss = self.loss_fn(y_prediction, y.long() if self.crossEntropy else y)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.iteration += 1
            
            if self.iteration % 100 == 0 and True:
                print(torch.mean(loss.data).item())
        return y_prediction.data

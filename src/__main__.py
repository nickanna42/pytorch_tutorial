## project imports
import torch as t
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn

## data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer_1_size = 16
        self.layer_2_size = 16
        self.linear1 = nn.Linear(28*28, self.layer_1_size) # layer 1
        self.linear2 = nn.Linear(self.layer_1_size, self.layer_2_size) # hidden layer 2
        self.final = nn.Linear(self.layer_2_size, 10) # output layer
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28) # image is 28x28, so flatten to 784
        x = self.relu(self.linear1(x)) # input to hidden layer 1
        x = self.relu(self.linear2(x)) # hidden layer 1 to hidden layer 2
        x = self.final(x) # hidden layer 2 to output layer
        return x
net = Net()

## loss function

entropy_f = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=0.001)
epochs = 10

## train the model
print('training the model...')
for epoch in range(epochs):
  net.train()
  for data in train_loader:
    X, y = data # X is the image, y is the label
    optimizer.zero_grad() # zero the gradients
    output = net(X.view(-1, 28*28)) # pass the image to the nueral network
    loss = entropy_f(output, y) # calculate the loss
    loss.backward() # backpropogate the loss 
    optimizer.step() # optimize the weights and biases
  print(f'epoch {epoch+1} of {epochs} complete')
## test the model

correct = 0
total = 0
print('testing the model...')
with t.no_grad(): # don't calculate the gradients
    for data in test_loader:
        x, y = data
        output = net(x.view(-1, 784))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct +=1
            total +=1
print(f'accuracy: {round(correct/total, 3)}')
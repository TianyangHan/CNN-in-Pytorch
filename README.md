# CNN-in-Pytorch
## Some basic codes and rules about CNN in the torch library

## Example 1
```python
BATCH_SIZE = 50    #50个batch
batch = 50           #一个batch有50张图片
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


#定义 CNN network的class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # inputshape(1, 28, 28)
       #第一个层cnn网络conv1
         nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters   out_features = filters
                kernel_size=5,              # filter size   （ 5 x 5 matrix ）
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2),  )
               # output shape (16, 28, 28) 
                   
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        
        #第二层cnn网络conv2
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)  参数与conv1的顺序是一样的，默认
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)   flatten 函数
        output = self.out(x)
        return output, x    # return x for visualization
```        
## The end of class model

```python
cnn = CNN()
print(cnn)  # net architecture
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters, lr = learning rate ,自己设置为LR
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted 损失函数

#开始主函数 training and testing
epoch = 50    #training times = 50  训练50遍
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # 这里的bx by是data的tuples， data = ( images, label )

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
```

## Example 2
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                train=True,
                transform=transforms.ToTensor(),
                download=True)

test_dataset = datasets.MNIST(root='./data/',
               train=False,
               transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=batch_size,
                     shuffle=False)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 输入1通道，输出10通道，kernel 5*5
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.mp = nn.MaxPool2d(2)
    # fully connect
    self.fc = nn.Linear(320, 10)

  def forward(self, x):
    # in_size = 64
    in_size = x.size(0) # one batch
    # x: 64*10*12*12
    x = F.relu(self.mp(self.conv1(x)))
    # x: 64*20*4*4
    x = F.relu(self.mp(self.conv2(x)))
    # x: 64*320
    x = x.view(in_size, -1) # flatten the tensor
    # x: 64*10
    x = self.fc(x)
    return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 200 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.data[0]))


def test():
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    # sum up batch loss
    test_loss += F.nll_loss(output, target, size_average=False).data[0]
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
  train(epoch)
  test()
```

值得注意的是：
整个过程中size的变化：

刚开始input的时候，x_size = 64 x 1 x a x a
进入卷基层后，x_size = 64 x 10 x b x b (a和b的关系的计算与kernel_size有关，具体可以百度，这里太懒，后面补上)
a b关系定义如下：
O=输出图像的尺寸。
I=输入图像的尺寸。
K=卷积层的核尺寸
N=核数量
S=移动步长
P =填充数
输出图像的通道数等于核数量N。
输出图像尺寸的计算公式如下：
![image](https://user-images.githubusercontent.com/89590789/130980426-bcf4d085-c86c-46f2-b43e-d29fdbeb6efc.png)


Relu层不改变形状
maxpooling操作使得 x_size = 64 x 10 x c x c
定义如下：
O=输出图像的尺寸。  
I=输入图像的尺寸。  
S=移动步长  
PS=池化层尺寸
不同于卷积层，池化层的输出通道数不改变
输出图像尺寸的计算公式如下：
![image](https://user-images.githubusercontent.com/89590789/130980480-4a4967b4-20e6-4ed7-b6f5-277c1faede97.png)


同理再进入第二层网络，size一样的变化
假设只经过一层conv2d，那么此时为x_size = 64 x 10 x c x c
经过flatten操作： x.view(batch_size,-1)或者x.view( x.size()[0],-1)，使得out_size = [64, 10 x (b/2) x (b/2)]

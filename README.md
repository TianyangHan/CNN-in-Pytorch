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


## Example 3
```python
import torch  
from torch.autograd import Variable  
import torch.nn as nn  
import matplotlib.pyplot as plt  
import torchvision  
import torch.utils.data as Data  
  
BATCH_SIZE = 64  
#学习率，学习率一般为0.01，0.1等等较小的数，为了在梯度下降求解时避免错过最优解  
LR = 0.001  
"""  
EPOCH 假如现在我有1000张训练图像，因为每次训练是64张，  
每当我1000张图像训练完就是一个EPOCH，训练多少个EPOCH自己决定  
"""  
DOWNLOAD_MNIST = False  
EPOCH = 1  
train_data = torchvision.datasets.MNIST(  
    root='./mnist',  
 train = True,  
 transform=torchvision.transforms.ToTensor(),  
 download=DOWNLOAD_MNIST  
)  
  
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 )  
#每个batch_size的shape为[64, 1, 28, 28]  
#测试集操作和上面注释一样  
test_data = torchvision.datasets.MNIST(  
    root='./mnist',  
 train = False,  
)  
  
  
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.0  
"""  
test_data.test_data中的shape为[10000, 28, 28]代表1w张图像，都是28x28，当时并未表明channels,因此在unsqueeze在1方向想加一个维度，  
则shape变为[10000, 1, 28, 28]，然后转化为tensor的float32类型，取1w张中的2000张，并且将其图片进行归一化处理，避免图像几何变换的影响  
"""  
#标签取前2000  
test_y = test_data.test_labels[:2000]  
  
#定义网络结构  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        #前面都是规定结构  
 #第一个卷积层，这里使用快速搭建发搭建网络 self.conv1 = nn.Sequential(  
            nn.Conv2d(  
                in_channels=1,#灰度图，channel为一  
 out_channels=16,#输出channels自己设定  
 kernel_size=3,#卷积核大小  
 stride=1,#步长  
 padding=1#padding=（kernel_size-stride）/2   往下取整  
 ),  
 nn.ReLU(),#激活函数，线性转意识到非线性空间  
 nn.MaxPool2d(kernel_size=2)#池化操作，降维，取其2x2窗口最大值代表此窗口，因此宽、高减半，channel不变  
 )  
        #此时shape为[16, 14, 14]  
 self.conv2 = nn.Sequential(  
            nn.Conv2d(  
                in_channels=16,  
 out_channels=32,  
 kernel_size=3,  
 stride=1,  
 padding=1  
 ),  
 nn.ReLU(),  
 nn.MaxPool2d(kernel_size=2)  
        )  
        #此时shape为[32, 7, 7]  
 #定义全连接层，十分类，并且全连接接受两个参数，因此为[32*7*7, 10] self.prediction = nn.Linear(32*7*7, 10)  
        #前向传播过程  
 def forward(self, x):  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(x.size(0), -1)  
        output = self.prediction(x)  
        return output  
  
#创建网络  
cnn = CNN()  
  
#大数据常用Adam优化器，参数需要model的参数，以及学习率  
optimizer = torch.optim.Adam(cnn.parameters(), LR)  
#定义损失函数，交叉熵  
loss_func = nn.CrossEntropyLoss()  
  
  
#训练阶段  
for epoch in range(EPOCH):  
    #step,代表现在第几个batch_size  
 #batch_x 训练集的图像 #batch_y 训练集的标签 for step, (batch_x, batch_y) in enumerate(train_loader):  
        #model只接受Variable的数据，因此需要转化  
 b_x = Variable(batch_x)  
        b_y = Variable(batch_y)  
        #将b_x输入到model得到返回值  
 output = cnn(b_x)  
        print(output)  
        #计算误差  
 loss = loss_func(output, b_y)  
        #将梯度变为0  
 optimizer.zero_grad()  
        #反向传播  
 loss.backward()  
        #优化参数  
 optimizer.step()  
        #打印操作，用测试集检验是否预测准确  
 if step % 50 == 0:  
            test_output = cnn(test_x)  
            #squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]  
 pre_y = torch.max(test_output, 1)[1].data.squeeze()  
            #总预测对的数除总数就是对的概率  
 accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))  
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" %accuracy)
```

## Example 4 ( Available )
```python
import torch  
from torch import nn,optim  
from torch.autograd import Variable  
from torch.utils.data import DataLoader  
from torchvision import datasets,transforms  
batch_size = 64  
learning_rate = 0.02  
num_eporches =20  
  
# 数据准备  
data_ft = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])  
  
train_dataset = datasets.MNIST("./data",train=True,transform=data_ft,download=False)  
test_dataset = datasets.MNIST("./data",train=False,transform=data_ft)  
  
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)  
  
  
# CNN模型  
# 建立四个卷积层网络 、两个池化层 、 1个全连接层  
# 第一层网络中，为卷积层，将28*28*1的图片，转换成16*26*26  
# 第二层网络中，包含卷积层和池化层。将16*26*26 -> 32*24*24,并且池化成cheng32*12*12  
# 第三层网络中，为卷积层，将32*12*12 -> 64*10*10  
# 第四层网络中，为卷积层和池化层，将 64*10*10 -> 128*8*8,并且池化成128*4*4  
# 第五次网络为全连接网络  
  
class CNN(nn.Module):  
    def __init__(self):  
        super(CNN, self).__init__()  
        self.layer1 = nn.Sequential(  
            nn.Conv2d(1, 16, kernel_size=3),  
 nn.BatchNorm2d(16),  
 nn.ReLU(inplace=True))  
        self.layer2 = nn.Sequential(  
            nn.Conv2d(16, 32, kernel_size=3),  
 nn.BatchNorm2d(32),  
 nn.ReLU(inplace=True),  
 nn.MaxPool2d(kernel_size=2, stride=2))  
        self.layer3 = nn.Sequential(  
            nn.Conv2d(32, 64, kernel_size=3),  
 nn.BatchNorm2d(64),  
 nn.ReLU(inplace=True))  
        self.layer4 = nn.Sequential(  
            nn.Conv2d(64, 128, kernel_size=3),  
 nn.BatchNorm2d(128),  
 nn.ReLU(inplace=True),  
 nn.MaxPool2d(kernel_size=2, stride=2))  
  
        self.fc = nn.Sequential(  
            nn.Linear(128 * 4 * 4, 1024),  
 nn.ReLU(inplace=True),  
 nn.Linear(1024, 128),  
 nn.ReLU(inplace=True),  
 nn.Linear(128, 10))  
  
    def forward(self, x):  
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        x = x.view(x.size(0), -1)  # 第二次卷积的输出拉伸为一行  
 x = self.fc(x)  
        return x  
  
  
model = CNN()  
if torch.cuda.is_available():  
    model = model.cuda()  
  
  
  
  
  
 #设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(),lr=learning_rate)  
  
  
  
  
  
  
# 训练模型  
epoch = 0  
for data in train_loader:  
    img, label = data  
  
    # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵  
 # img = img.view(img.size(0), -1)  
 if torch.cuda.is_available():  
        img = img.cuda()  
        label = label.cuda()  
    else:  
        img = Variable(img)  
  
        label = Variable(label)  
  
    out = model(img)  
    loss = criterion(out, label)  
    print_loss = loss.data.item()  
  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    epoch += 1  
 if epoch % 50 == 0:  
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
		
		
		
		
# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    
    # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
    #img = img.view(img.size(0), -1)
    
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
 
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))
```		

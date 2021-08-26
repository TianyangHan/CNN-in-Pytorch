# CNN-in-Pytorch
Some basic codes and rules about CNN in the torch library

Example 1
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
        
# The end of class model

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

import torch
import numpy as np
import torch.nn as nn
from utils.read_data import read_dataset
from utils.ResNet import ResNet18
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 读数据
batch_size = 128
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, img_path='data')

# 加载模型（使用预处理模型，仅修改最后一层）
num_class = 10
model = ResNet18()
"""
ResNet18网络的7x7卷积层和max池化操作容易丢失一部分信息,
所以在此我们将7x7的卷积层和max池化层去掉,替换为一个3x3的卷积层,
同时减小该卷积层的步长和填充大小
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, num_class)
model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)
lr = []
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# 使用CLR调度器
step_size_up = len(train_loader) * 2  # 设置学习率上升的步长
clr_scheduler = CyclicLR(optimizer, base_lr=0.00045, max_lr=0.008, step_size_up=step_size_up)
# Start Training
epochs = 250
valid_loss_min = np.Inf
accuracy = []

counter = 0
for epoch in range(1, epochs+1):
    # 训练中验证集损失
    train_loss = 0.0
    valid_loss = 0.0
    total_samples = 0
    correct_samples = 0

    # 训练集的模型
    model.train(True)  # 作用是启用batch normalization和drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data).to(device)
        # 计算损失
        loss = criterion(output, target)
        # 后向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        clr_scheduler.step()
        # 更新损失
        train_loss += loss.item() * data.size(0)
    # 验证集的模型(同理)

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data).to(device)

        loss = criterion(output, target)

        valid_loss += loss.item() * data.size(0)

        # 将输出概率转换为预测标签
        _, predicted = torch.max(output, 1)
        # 将预测标签与实际标签比较
        correct = predicted.eq(target.data.view_as(predicted))

        total_samples += batch_size
        for i in correct:
            if i:
                correct_samples += 1

    # 计算平均损失
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # 显示损失函数
    print('Epoch: {}/{} \tTraining Loss: {:.12f} \tValidation Loss: {:.12f}\n'
          .format(epoch, epochs, train_loss, valid_loss))
    # 计算准确率
    print("Epoch: {}/{} \tAccuracy is:".format(epoch, epochs), 100 * correct_samples / total_samples, "%\n")
    accuracy.append(correct_samples / total_samples)

    # 若验证集损失函数减少，则保存模型
    if valid_loss <= valid_loss_min:
        print('valid_loss decreased: {:.6f} -> {:.6f} \tSaving model...'
              .format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), './checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        print('valid_loss changed : {:.6f} -> {:.6f} \tNot saving model...'.format(valid_loss_min, valid_loss))
        counter += 1

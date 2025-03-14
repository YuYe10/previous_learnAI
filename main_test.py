import torch
import torch.nn as nn
from utils.read_data import read_dataset
from utils.ResNet import ResNet18
# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 100
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, img_path='data')
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()  # 验证模型
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data).to(device)
    # 将输出概率转换为预测类
    _, pred = torch.max(output, 1)
    # 将预测与真实标签进行比较
    correct_tensor = pred.eq(target.data.view_as(pred))
    total_sample += batch_size
    for i in correct_tensor:
        if i:
            right_sample += 1
print("Accuracy:", 100*right_sample/total_sample, "%")

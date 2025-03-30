import torch
from torchvision import transforms
from PIL import Image
from model import ResNet18

# 加载模型
model = ResNet18()
model.load_state_dict(torch.load('models/best_model.pth'))

# 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 定义类别
classes = ["butterfly",
               "cat",
               "chicken",
               "cow",
               "dog",
               "elephant",
               "horse",
               "rangno",
               "sheep",
               "squirrel"
               ]

# 打开图片
image = Image.open('904.jpeg')

# 定义归一化参数和数据处理方法
normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155, 0.06216329, 0.05930814])
test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

# 处理图片
image = test_transform(image)

# 添加批次维度
image = image.unsqueeze(0)

# 进行预测
with torch.no_grad():
    model.eval()
    image = image.to(device)
    output = model(image)
    pre_lab = torch.argmax(output, dim=1)
    result = pre_lab.item()

# 输出预测结果
print("预测值：", classes[result])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg19

# 加载Style和Content图片
style_image_path = 'D:\\桌面\\style_image.jpg'
content_image_path = 'D:\\桌面\\content_image.jpg'
style_image = Image.open(style_image_path).convert('RGB')
content_image = Image.open(content_image_path).convert('RGB')
# 将图片转换为张量
style_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
content_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
style_image_tensor = style_transform(style_image)
content_image_tensor = content_transform(content_image)
# 确保图片张量是可变的，以便在反向传播期间进行修改
style_image_tensor = Variable(style_image_tensor.unsqueeze(0), requires_grad=True)
content_image_tensor = Variable(content_image_tensor.unsqueeze(0), requires_grad=True)


# 定义生成器和判别器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(content_image_tensor.shape[1] * content_image_tensor.shape[2], 1024),
            nn.ReLU(True),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )


# 加载预训练的VGG网络
def load_vgg():
    vgg = vgg19(pretrained=True).features
    # 去掉VGG网络的最后一层分类器，保留前面的特征提取部分
    vgg = nn.Sequential(*list(vgg.children())[:-1])
    return vgg


# 预处理图像
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)
    return image


# 反向预处理图像
def deprocess_image(image):
    transform = transforms.Compose([
        transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    ])
    image = transform(image.squeeze(0))
    image = torch.clamp(image, 0, 1)
    return image


# 预处理图像
style_image = preprocess_image(style_image)
content_image = preprocess_image(content_image)
# 实例化生成器和判别器
generator = Generator()
discriminator = Discriminator()
# 加载VGG网络
vgg = load_vgg()
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)
# 训练GAN网络模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    generated_image = generator(content_image)
    # 计算生成图像的风格和内容特征
    generated_features = vgg(generated_image)
    content_features = vgg(content_image)
    style_features = vgg(style_image)

    # 计算生成图像的风格损失和内容损失
    style_loss = criterion(generated_features['style'], style_features['style'])
    content_loss = criterion(generated_features['content'], content_features['content'])
    total_loss = style_loss + content_loss
    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss.item()))
# 反向预处理图像


style_image = deprocess_image(style_image)
content_image = deprocess_image(content_image)
# 展示style_image图片、content_image图片和风格迁移后生成的图片
import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.imshow(style_image)
plt.title("Style Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(content_image)
plt.title("Content Image")
plt.axis("off")
plt.subplot(1, 3, 3)


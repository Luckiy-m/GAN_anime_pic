# GAN动漫头像生成

基于DCGAN（Deep Convolutional Generative Adversarial Network）实现的动漫头像生成项目。

## 文件结构

- [model.py](file:///workspace/model.py) - 包含生成器和判别器网络结构
- [dataset.py](file:///workspace/dataset.py) - 数据加载和预处理模块
- [train.py](file:///workspace/train.py) - 训练脚本
- [generate.py](file:///workspace/generate.py) - 图像生成和可视化脚本
- [requirements.txt](file:///workspace/requirements.txt) - 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

将动漫头像图片数据集放在 `./data` 目录下，目录结构如下：

```
./data/
    └── anime_faces/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

## 训练模型

运行训练脚本：

```bash
python train.py
```

训练过程中会自动创建：
- `./checkpoints/` - 保存模型检查点
- `./results/` - 保存训练过程生成的图像和损失曲线

## 生成图像

使用训练好的模型生成动漫头像：

```bash
# 生成图像
python generate.py --checkpoint ./checkpoints/generator_epoch_100.pth --num_images 64

# 图像插值
python generate.py --mode interpolate --checkpoint ./checkpoints/generator_epoch_100.pth --num_images 10
```

## 网络结构

### 生成器 (Generator)
- 输入：100维随机向量
- 输出：64×64×3的RGB图像
- 使用转置卷积层进行上采样
- 激活函数：ReLU（除最后一层使用Tanh）

### 判别器 (Discriminator)
- 输入：64×64×3的RGB图像
- 输出：单个标量（0-1之间的概率值）
- 使用卷积层进行下采样
- 激活函数：LeakyReLU（除最后一层使用Sigmoid）

## 超参数

- 学习率：0.0002
- 批量大小：128
- 潜在维度：100
- 训练轮数：100
- 优化器：Adam（beta1=0.5）
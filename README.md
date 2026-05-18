# GAN动漫头像生成

基于DCGAN（Deep Convolutional Generative Adversarial Network）实现的动漫头像生成项目，包含完整的数据抓取、模型训练和图像生成流程。

## 文件结构

- [model.py](file:///workspace/model.py) - 包含生成器和判别器网络结构
- [dataset.py](file:///workspace/dataset.py) - 数据加载和预处理模块
- [train.py](file:///workspace/train.py) - 训练脚本
- [generate.py](file:///workspace/generate.py) - 图像生成和可视化脚本
- [scraper.py](file:///workspace/scraper.py) - 动漫头像爬虫脚本
- [pipeline.py](file:///workspace/pipeline.py) - 完整流程脚本（爬虫→训练→生成）
- [requirements.txt](file:///workspace/requirements.txt) - 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 一键执行完整流程

```bash
# 抓取500张图像，训练100轮，生成图像
python pipeline.py full --num_images 500 --num_epochs 100
```

### 分步执行

#### 1. 抓取数据

```bash
# 使用内置图像生成器（适合测试）
python scraper.py --num_images 500

# 或使用完整流程脚本
python pipeline.py scrape --num_images 500
```

#### 2. 训练模型

```bash
python train.py

# 或使用完整流程脚本
python pipeline.py train
```

#### 3. 生成图像

```bash
python generate.py --checkpoint ./checkpoints/generator_epoch_100.pth --num_images 64

# 或使用完整流程脚本
python pipeline.py generate --checkpoint ./checkpoints/generator_epoch_100.pth
```

## 高级用法

### pipeline.py 完整流程控制

```bash
# 跳过某些步骤
python pipeline.py full --skip_scrape  # 已准备好数据，跳过抓取
python pipeline.py full --skip_train   # 已有模型，跳过训练
python pipeline.py full --skip_generate  # 只训练不生成

# 单独执行各个步骤
python pipeline.py scrape --num_images 1000  # 只抓取数据
python pipeline.py train                    # 只训练模型
python pipeline.py generate --num_images 128  # 只生成图像
```

### generate.py 图像生成

```bash
# 图像插值
python generate.py --mode interpolate --num_images 10

# 自定义保存路径
python generate.py --save_path ./output/my_images.png
```

## 数据集准备

### 使用爬虫自动获取

内置爬虫支持多种数据源：

```bash
# 使用 placeholder 生成器（默认，适合测试）
python scraper.py --source placeholder --num_images 500

# 使用 thisanimedoesnotexist.ai 网站
python scraper.py --source thisanimedoesnotexist --num_images 500
```

### 手动准备数据集

将动漫头像图片数据集放在 `./data/anime_faces/` 目录下：

```
./data/
    └── anime_faces/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

## 训练过程

运行训练脚本时会自动创建：
- `./checkpoints/` - 每个epoch保存模型检查点
- `./results/` - 保存训练过程生成的图像和损失曲线

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
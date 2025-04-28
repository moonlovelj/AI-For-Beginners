import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TABLEAU_COLORS

def visualize_detections(image, boxes, classes, scores, figsize=(12, 12)):
    """
    在图像上可视化检测结果
    
    参数:
        image: 图像
        boxes: 边界框坐标 [x1, y1, x2, y2]
        classes: 检测到的类别
        scores: 检测到的置信度
        figsize: 图像显示大小
    """
    # 创建图像
    plt.figure(figsize=figsize)
    plt.imshow(image)
    
    # 获取颜色
    colors = list(TABLEAU_COLORS.values())
    
    # 获取图像尺寸
    img_height, img_width = image.shape[:2]
    
    # 绘制每个检测结果
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        # 解析边界框坐标
        x1, y1, x2, y2 = box
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 计算宽度和高度
        width = x2 - x1
        height = y2 - y1
        
        # 跳过无效的边界框
        if width <= 0 or height <= 0:
            continue
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 绘制矩形
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # 添加标签
        plt.text(
            x1, y1 - 5,
            f"{cls}: {score:.2f}",
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8)
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dataset_samples(dataset, num_samples=5, figsize=(15, 10), random_seed=None):
    """
    随机可视化数据集中的num_samples张图片及其标注
    
    参数:
        dataset: HollywoodHeadsDataset实例
        num_samples: 要可视化的样本数
        figsize: 图像显示大小
        random_seed: 随机种子，设置后可重现相同的随机选择结果
    """
    # 设置随机种子以便结果可重现
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 从数据集中随机选择样本索引
    dataset_size = len(dataset)
    if dataset_size <= num_samples:
        # 如果数据集小于请求的样本数，使用所有样本
        sample_indices = list(range(dataset_size))
    else:
        # 随机选择不重复的样本索引
        sample_indices = np.random.choice(dataset_size, num_samples, replace=False)
    
    print(f"随机选择的样本索引: {sample_indices}")
    
    # 处理每个选定的样本
    for idx, sample_idx in enumerate(sample_indices):
        # 获取原始样本（未经过变换）用于可视化
        image, annotations = dataset.get_raw_sample(sample_idx)
        
        # 获取边界框和标签
        boxes = annotations['boxes'].numpy() if isinstance(annotations['boxes'], torch.Tensor) else annotations['boxes']
        labels = annotations['labels'].numpy() if isinstance(annotations['labels'], torch.Tensor) else annotations['labels']
        
        # 创建分数数组 (标注不包含分数，设为1.0)
        scores = np.ones(len(labels))
        
        # 转换类别ID为类别名称
        class_names = [f"Head" for _ in labels]  # 假设所有标注都是头部
        
        # 使用visualize_detections函数可视化
        print(f"Sample {idx+1} (Dataset index: {sample_idx}):")
        visualize_detections(
            image, 
            boxes, 
            class_names, 
            scores, 
            figsize=figsize
        )

class HollywoodHeadsDataset(Dataset):
    def __init__(self, root_dir, train=True, train_ratio=0.8, max_samples=None, transform=None):
        """
        初始化HollywoodHeads数据集
        
        参数:
            root_dir (string): 数据集根目录，包含JPEGImages和Annotations两个子文件夹
            train (bool): 如果为True，返回训练集；否则返回验证集
            train_ratio (float): 用于训练的数据比例，默认0.8
            max_samples (int, optional): 最大使用的样本数量，None表示使用全部
            transform (callable, optional): 用于图像预处理的变换
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        # 获取所有XML文件路径
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        
        self.annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        
        # 如果指定了max_samples，限制使用的样本数量
        if max_samples is not None:
            self.annotation_files = self.annotation_files[:max_samples]
        
        # 划分训练集和验证集
        n_samples = len(self.annotation_files)
        n_train = int(train_ratio * n_samples)
        
        if train:
            self.annotation_files = self.annotation_files[:n_train]
        else:
            self.annotation_files = self.annotation_files[n_train:]

        # 构建有效的图像和标注路径对
        valid_pairs = []
        for ann_file in self.annotation_files:
            ann_path = os.path.join(self.annotations_dir, ann_file)
            
            # 尝试不同的图像扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                img_file = ann_file.replace('.xml', ext)
                img_path = os.path.join(self.images_dir, img_file)
                if os.path.exists(img_path):
                    valid_pairs.append((img_path, ann_path))
                    break

        # 使用验证过的路径对
        self.image_paths = [pair[0] for pair in valid_pairs]
        self.annotation_paths = [pair[1] for pair in valid_pairs]
        
        print(f"Loaded {'training' if train else 'validation'} dataset with {len(self.image_paths)} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_raw_sample(self, idx):
        """
        获取未经任何变换的原始样本，用于可视化
        
        参数:
            idx: 样本索引
            
        返回:
            tuple: (原始图像, 标注字典)
        """
        # 获取图像路径和注释路径
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            # 返回一个简单的替代样本
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            annotations = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }
            return image, annotations
        
        # 转换BGR到RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 解析注释文件
        try:
            annotations = self.parse_annotation(annotation_path)
        except Exception as e:
            print(f"Error parsing annotation {annotation_path}: {e}")
            # 如果解析失败，创建一个空标签
            annotations = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }
        
        return image, annotations
    
    def __getitem__(self, idx):
        try:
            # 获取原始图像和标注
            image, annotations = self.get_raw_sample(idx)
            
            # 创建目标字典 - 仅包含torchvision模型所需的字段
            target = {
                'boxes': annotations['boxes'],
                'labels': annotations['labels']
            }
            
            # 应用变换（如果有）
            if self.transform is not None and isinstance(image, np.ndarray):
                image = self.transform(image)
            
            return image, target
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个简单的替代样本
            return np.zeros((100, 100, 3), dtype=np.uint8), {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }
    
    def parse_annotation(self, annotation_path):
        # 读取 XML 文件
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # 检查元素是否存在
            name_elem = obj.find('name')
            if name_elem is None:
                continue  # 跳过缺少名称的对象
                
            # 使用 .text 前检查元素是否存在
            name = name_elem.text
            
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue  # 跳过缺少边界框的对象
                
            # 为每个坐标值添加类似的检查
            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')
            
            # 确保所有元素都存在
            if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                continue
                
            # 安全获取坐标值
            try:
                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
            except (ValueError, AttributeError):
                continue  # 跳过无法解析的坐标
                
            # 确保坐标有效 (x1 < x2, y1 < y2)
            if xmin >= xmax or ymin >= ymax:
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # 假设所有对象都是同一类
        
        # 将结果转换为张量
        if not boxes:
            # 如果没有找到有效的边界框，返回空数组
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64)
            }
                    
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

# 用于创建标准化变换的辅助函数
def get_transform(train=True):
    """
    创建图像变换管道
    
    参数:
        train: 是否为训练模式，训练模式会添加数据增强
        
    返回:
        transforms.Compose对象
    """
    transforms_list = []
    
    # 转换为PyTorch张量
    transforms_list.append(transforms.ToTensor())
    
    # 标准化 - 使用ImageNet预训练模型的均值和标准差
    transforms_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    # 训练时添加数据增强
    if train:
        transforms_list.insert(0, transforms.ToPILImage())  # 需要先转换为PIL图像
        transforms_list.insert(1, transforms.RandomHorizontalFlip(0.5))
    
    return transforms.Compose(transforms_list)

# 示例：如何使用带有变换的数据集
def create_dataloaders(root_dir, train_batch_size=8, val_batch_size=1, max_samples=None):
    """
    创建训练和验证数据加载器
    
    参数:
        root_dir: 数据集根目录
        train_batch_size: 训练批大小
        val_batch_size: 验证批大小
        
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 定义自定义collate_fn，用于处理不同大小的图像
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # 创建数据集
    train_dataset = HollywoodHeadsDataset(
        root_dir=root_dir,
        train=True,
        transform=get_transform(train=True),
        max_samples=max_samples
    )
    
    val_dataset = HollywoodHeadsDataset(
        root_dir=root_dir,
        train=False,
        transform=get_transform(train=False),
        max_samples=max_samples
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataset, train_loader, val_dataset, val_loader
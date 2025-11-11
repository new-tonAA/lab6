import sys
import os
import torch
import argparse
from torchvision import transforms

# ----------------------------
# 设置项目根目录，让 Python 能找到 tools/models 等模块
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# ----------------------------
# 导入项目模块
# ----------------------------
from tools.imagedb import FaceDataset
from models.rnet import RNet
from training.rnet.trainer import RNetTrainer
from training.rnet.config import Config as RNetConfig
from tools.logger import Logger
from checkpoint import CheckPoint

# ----------------------------
# 获取配置
# ----------------------------
cfg = RNetConfig()
if not os.path.exists(cfg.save_path):
    os.makedirs(cfg.save_path)

# ----------------------------
# 设置设备和随机种子
# ----------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
use_cuda = cfg.use_cuda and torch.cuda.is_available()
torch.manual_seed(cfg.manualSeed)
torch.cuda.manual_seed(cfg.manualSeed)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# ----------------------------
# 设置 DataLoader
# ----------------------------
kwargs = {'num_workers': cfg.nThreads, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = FaceDataset(cfg.annoPath, transform=transform, is_train=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batchSize,
    shuffle=True,
    **kwargs
)

# ----------------------------
# 设置模型
# ----------------------------
model = RNet().to(device)

# ----------------------------
# 设置 checkpoint 和优化器
# ----------------------------
checkpoint = CheckPoint(cfg.save_path)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.step, gamma=0.1)

# ----------------------------
# 设置训练器和日志
# ----------------------------
logger = Logger(cfg.save_path)
trainer = RNetTrainer(cfg.lr, train_loader, model, optimizer, scheduler, logger, device)

# ----------------------------
# 开始训练
# ----------------------------
for epoch in range(1, cfg.nEpochs + 1):
    cls_loss, box_loss, total_loss, accuracy = trainer.train(epoch)
    print(f"Epoch {epoch} - cls_loss: {cls_loss:.4f}, box_loss: {box_loss:.4f}, total_loss: {total_loss:.4f}, accuracy: {accuracy:.4f}")
    checkpoint.save_model(model, index=epoch)
    scheduler.step()

import sys
sys.path.append('./')

import os
import torch
import pandas as pd
from torchvision import transforms
from tools.imagedb import FaceDataset
from models.pnet import PNet
from training.pnet.trainer import PNetTrainer
from training.pnet.config import Config
from tools.logger import Logger
from checkpoint import CheckPoint

# ------------------- 配置 -------------------
config = Config()
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

# ------------------- 设备 -------------------
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
use_cuda = config.use_cuda and torch.cuda.is_available()
torch.manual_seed(config.manualSeed)
torch.cuda.manual_seed(config.manualSeed)
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# ------------------- 数据加载 -------------------
kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_loader = torch.utils.data.DataLoader(
    FaceDataset(config.annoPath, transform=transform, is_train=True),
    batch_size=config.batchSize,
    shuffle=True,
    **kwargs
)

# ------------------- 模型与优化器 -------------------
model = PNet().to(device)
checkpoint = CheckPoint(config.save_path)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

logger = Logger(config.save_path)
trainer = PNetTrainer(config.lr, train_loader, model, optimizer, scheduler, logger, device)

# ------------------- 训练循环 -------------------
for epoch in range(1, config.nEpochs + 1):
    print(f"Starting Epoch {epoch}/{config.nEpochs}...")

    # train() 返回本 epoch 平均值，内部已经执行 optimizer.step()
    cls_loss_, box_offset_loss, total_loss, accuracy = trainer.train(epoch)

    # 在 optimizer.step() 后更新学习率，避免 warning
    scheduler.step()

    # 保存当前 epoch 模型
    checkpoint.save_model(model, index=epoch)

    # 保存 Excel 文件，每个 epoch 一个
    epoch_records = [{
        'Epoch': epoch,
        'ClsLoss': cls_loss_,
        'BoxLoss': box_offset_loss,
        'TotalLoss': total_loss,
        'Accuracy': accuracy
    }]
    excel_path = os.path.join(config.save_path, f'train_epoch_{epoch:03d}.xlsx')
    pd.DataFrame(epoch_records).to_excel(excel_path, index=False)
    print(f"Epoch {epoch} log saved to {excel_path}")

print("All training finished.")



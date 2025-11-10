
# 人脸检测

实验指导 https://www.zybuluo.com/mymy/note/1621354

文件结构


注意：部分大文件不在github，记得去微信找  
mtcnn_pytorch/  
├── annotations/下面一堆txt（将annotations压缩包解压放到这里）  
├── data/（把12、24、48文件夹放到这里）  
├── models/  
├── preprocessing/  
├── result/（训练结果）  
│   └── pnet/  
│       └── log_bs512_lr0.010_072402/  
├── tools/  
├── train_data/（pnet生成12x12结果）  
├── training/  
├── .gitignore  
├── README.md  
├── checkpoint.py  
├── config.py  
├── test_image.py  
├── test_on_FDDB.py  
├── test_youModel_images.py  


zyf已做步骤：实现PNet的50个epoch (2025/11/9 - 2025/11/10)
1. 运行`split_wider.py`，实现annotations中两个txt文件的合并（项目中已有结果）
2. 将数据集http://shuoyang1213.me/WIDERFACE/ 下载后解压到与lab4**同级**（也就是mtcnn_pytorch与lab4同级）
3. 运行`gen_pnet_data.py`，实现生成训练样本：- PNet 样本：随机裁剪图像 patches（含人脸 / 非人脸，比例 1:3），尺寸 12×12（运行注意文件，  且项目中没有放运行结果，太大了），运行结果将出现在train_data中的12文件夹
4. 运行`assemble_pnet.py`
5. 将4中的12文件夹移动到data文件夹下
6. 运行pnet的train.py
7. 运行结果放在results/pnet文件夹中

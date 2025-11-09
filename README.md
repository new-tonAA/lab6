
# 人脸检测

实验指导 https://www.zybuluo.com/mymy/note/1621354

zyf已做步骤：实现PNet的20个epoch
1. 运行`split_wider.py`，实现annotations中两个txt文件的合并（项目中已有结果）
2. 将数据集http://shuoyang1213.me/WIDERFACE/ 下载后解压到与lab4**同级**（也就是mtcnn_pytorch与lab4同级）
3. 运行`gen_pnet_data.py`，实现生成训练样本：- PNet 样本：随机裁剪图像 patches（含人脸 / 非人脸，比例 1:3），尺寸 12×12（运行注意文件，  且项目中没有放运行结果，太大了），运行结果将出现在training中的12文件夹
4. 

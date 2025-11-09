import os
import random

def split_wider_dataset(annotation_file, train_file, val_file, val_ratio=0.2):
    """
    将 wider_origin_anno.txt 按比例划分为训练集和验证集
    每行格式: image_path x1 y1 x2 y2 le_x le_y re_x re_y n_x n_y lm_x lm_y rm_x rm_y ...
    """
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    total = len(lines)
    val_count = int(total * val_ratio)
    train_lines = lines[val_count:]
    val_lines = lines[:val_count]

    def parse_line(line):
        parts = line.strip().split()
        img_path = parts[0]
        numbers = list(map(float, parts[1:]))
        # 每14个数字代表1张人脸
        face_count = len(numbers) // 14
        return img_path, face_count

    # 输出统计信息
    total_faces = 0
    for line in lines:
        _, count = parse_line(line)
        total_faces += count
    print(f"📊 总图片数: {len(lines)}, 总人脸数: {total_faces}")
    print(f"📚 训练集: {len(train_lines)}, 验证集: {len(val_lines)}")

    # 保存文件
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    with open(val_file, 'w') as f:
        f.writelines(val_lines)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # annotations 目录在脚本所在目录的上一级
    base_dir = os.path.join(script_dir, '..', 'annotations')

    annotation_file = os.path.join(base_dir, 'wider_origin_anno.txt')
    train_file = os.path.join(base_dir, 'wider_train.txt')
    val_file = os.path.join(base_dir, 'wider_val.txt')

    split_wider_dataset(annotation_file, train_file, val_file)

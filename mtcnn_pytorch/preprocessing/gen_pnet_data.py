import sys
import os
import argparse
import numpy as np
import cv2
import numpy.random as npr

# 添加项目根目录到模块搜索路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

from tools.utils import IoU
import config


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def gen_pnet_data(data_dir, anno_file, prefix):
    # 构造输出目录
    neg_save_dir = os.path.join(data_dir, "12", "negative")
    pos_save_dir = os.path.join(data_dir, "12", "positive")
    part_save_dir = os.path.join(data_dir, "12", "part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir, config.ANNO_STORE_DIR]:
        ensure_dir(dir_path)

    # 输出文件路径
    post_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_POSTIVE_ANNO_FILENAME)
    neg_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_NEGATIVE_ANNO_FILENAME)
    part_save_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_PART_ANNO_FILENAME)

    f1 = open(post_save_file, 'w')
    f2 = open(neg_save_file, 'w')
    f3 = open(part_save_file, 'w')

    # 读取注释文件
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    print(f"{len(annotations)} pics in total")

    p_idx = n_idx = d_idx = idx = box_idx = 0

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        img_rel = annotation[0]

        # 构造绝对图片路径，兼容 / 或 \
        img_path = os.path.normpath(os.path.join(prefix, img_rel))
        if not os.path.exists(img_path):
            img_path_alt = os.path.normpath(os.path.join(prefix, img_rel.replace('\\', '/')))
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
        if not os.path.exists(img_path):
            print(f"警告：找不到图片 - {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 - {img_path}")
            continue

        bbox = np.array(list(map(float, annotation[1:])), dtype=np.int32).reshape(-1, 4)
        height, width, channel = img.shape
        idx += 1
        if idx % 100 == 0:
            print(f"{idx} images done")

        # 生成负样本
        neg_num = 0
        while neg_num < 50:
            size = npr.randint(12, min(width, height) // 2)
            nx, ny = npr.randint(0, width - size), npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            Iou = IoU(crop_box, bbox)
            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg")
                f2.write(save_file + ' 0\n')
                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12))
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        # 遍历每个bbox生成正样本和part样本
        for box in bbox:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # 生成负样本（与gt有一定重叠）
            for _ in range(5):
                size = npr.randint(12, min(width, height) // 2)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1, ny1 = max(0, x1 + delta_x), max(0, y1 + delta_y)
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, bbox)
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg")
                    cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                    resized_im = cv2.resize(cropped_im, (12, 12))
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # 正样本和part样本
            for _ in range(20):
                size = npr.randint(int(min(w, h) * 0.8), int(np.ceil(1.25 * max(w, h))))
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2, ny2 = nx1 + size, ny1 + size
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                offset_x1, offset_y1 = (x1 - nx1) / size, (y1 - ny1) / size
                offset_x2, offset_y2 = (x2 - nx2) / size, (y2 - ny2) / size
                box_ = box.reshape(1, -1)
                iou_val = IoU(crop_box, box_)
                if iou_val >= 0.65:
                    save_file = os.path.join(pos_save_dir, f"{p_idx}.jpg")
                    f1.write(save_file + f' 1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n')
                    cv2.imwrite(save_file, cv2.resize(img[ny1:ny2, nx1:nx2, :], (12, 12)))
                    p_idx += 1
                elif iou_val >= 0.4:
                    save_file = os.path.join(part_save_dir, f"{d_idx}.jpg")
                    f3.write(save_file + f' -1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n')
                    cv2.imwrite(save_file, cv2.resize(img[ny1:ny2, nx1:nx2, :], (12, 12)))
                    d_idx += 1
            box_idx += 1

        print(f"{idx} images done, pos: {p_idx}, part: {d_idx}, neg: {n_idx}")

    f1.close()
    f2.close()
    f3.close()


def parse_args():# 记得更换路径
    parser = argparse.ArgumentParser(description='generate pnet training data')

    # 训练数据输出目录（你可以改）
    parser.add_argument(
        '--face_traindata_store',
        dest='traindata_store',
        default=r"E:\MachingLearningEx\lab4\lab4\mtcnn_pytorch\train_data"
    )

    # wider_origin_anno.txt 的绝对路径
    parser.add_argument(
        '--anno_file',
        dest='annotation_file',
        default=r"E:\MachingLearningEx\lab4\lab4\mtcnn_pytorch\annotations\wider_origin_anno.txt"
    )

    # WIDER_train/images 绝对路径
    parser.add_argument(
        '--prefix_path',
        dest='prefix_path',
        default=r"E:\MachingLearningEx\lab4\WIDER_train\images"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gen_pnet_data(args.traindata_store, args.annotation_file, args.prefix_path)
    

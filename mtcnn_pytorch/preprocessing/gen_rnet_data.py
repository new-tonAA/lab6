import sys
import os
import argparse
import pickle
import time
import numpy as np
import cv2

# 添加项目根目录到模块搜索路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

import tools.vision as vision
import config
from tools.train_detect import MtcnnDetector
from tools.imagedb import ImageDB
from tools.image_reader import TestImageLoader
from tools.utils import IoU, convert_to_square

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path='', use_cuda=True, vis=False):
    mtcnn_detector = MtcnnDetector(
        p_model_path=pnet_model_file,
        r_model_path=None,
        o_model_path=None,
        min_face_size=12,
        use_cuda=use_cuda
    )
    device = mtcnn_detector.device

    imagedb = ImageDB(anno_file, mode="test", prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)

    all_boxes = []
    batch_idx = 0

    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print(f"{batch_idx} images done")
        im = databatch
        t = time.time()
        boxes, boxes_align = mtcnn_detector.detect_pnet(im)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            continue
        if vis:
            vision.vis_face(im, boxes_align)
        t1 = time.time() - t
        print(f'time cost for image {batch_idx} / {image_reader.size} : {t1:.4f}')
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = config.TRAIN_DATA_DIR
    ensure_dir(save_path)

    save_file = os.path.join(save_path, f"pnet_detections_{int(time.time())}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    get_rnet_sample_data(data_dir, anno_file, save_file, prefix_path)


def get_rnet_sample_data(data_dir, anno_file, det_boxes_file, prefix_path):
    neg_save_dir = os.path.join(data_dir, "24", "negative")
    pos_save_dir = os.path.join(data_dir, "24", "positive")
    part_save_dir = os.path.join(data_dir, "24", "part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        ensure_dir(dir_path)

    # load ground truth
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 24
    im_idx_list, gt_boxes_list = [], []
    print(f"processing {len(annotations)} images in total")

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix_path, annotation[0])
        im_idx = os.path.normpath(im_idx)  # Windows 兼容 / 和 \
        if not os.path.exists(im_idx):
            im_idx_alt = im_idx.replace('\\', '/')
            if os.path.exists(im_idx_alt):
                im_idx = im_idx_alt
        if not os.path.exists(im_idx):
            print(f"警告：找不到图片 - {im_idx}")
            continue
        boxes = np.array(list(map(float, annotation[1:])), dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = config.ANNO_STORE_DIR
    ensure_dir(save_path)
    f1 = open(os.path.join(save_path, f'pos_{image_size}.txt'), 'w')
    f2 = open(os.path.join(save_path, f'neg_{image_size}.txt'), 'w')
    f3 = open(os.path.join(save_path, f'part_{image_size}.txt'), 'w')

    with open(det_boxes_file, 'rb') as det_handle:
        det_boxes = pickle.load(det_handle)

    print(len(det_boxes), len(im_idx_list))
    assert len(det_boxes) == len(im_idx_list), "incorrect detections or ground truths"

    n_idx = p_idx = d_idx = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        cur_n_idx = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width, height = x_right - x_left, y_bottom - y_top
            if width < 20 or x_left <= 0 or y_top <= 0 or x_right >= img.shape[1] or y_bottom >= img.shape[0]:
                continue
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom, x_left:x_right, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size))

            if np.max(Iou) < 0.3:
                cur_n_idx += 1
                if cur_n_idx <= 50:
                    save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg")
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
            else:
                idx_max = np.argmax(Iou)
                assigned_gt = gts[idx_max]
                x1, y1, x2, y2 = assigned_gt
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, f"{p_idx}.jpg")
                    f1.write(save_file + f' 1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n')
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, f"{d_idx}.jpg")
                    f3.write(save_file + f' -1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n')
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

    f1.close()
    f2.close()
    f3.close()


def parse_args():
    parser = argparse.ArgumentParser(description='generate rnet training data')
    parser.add_argument('--face_traindata_store', dest='traindata_store',
                        default=r"E:\MachingLearningEx\lab4\lab4\mtcnn_pytorch\train_data") # 记得改路径
    parser.add_argument('--anno_file', dest='annotation_file',
                        default=r"E:\MachingLearningEx\lab4\lab4\mtcnn_pytorch\annotations\wider_origin_anno.txt") # 记得改路径
    parser.add_argument('--pmodel_file', dest='pnet_model_file',
                        default=r"E:\MachingLearningEx\lab4\lab4\mtcnn_pytorch\results\pnet\log_bs512_lr0.010_072402\check_point\model_050.pth") # 记得改路径
    parser.add_argument('--gpu', dest='use_cuda', default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='prefix_path',
                        default=r"E:\MachingLearningEx\lab4\WIDER_train\images") # 记得改路径
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    gen_rnet_data(args.traindata_store, args.annotation_file,
                  args.pnet_model_file, args.prefix_path, args.use_cuda)

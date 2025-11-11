import sys
import os

# 把项目根目录加入模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import config
import preprocessing.assemble as assemble

if __name__ == '__main__':
    anno_list = []

    rnet_postive_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSTIVE_ANNO_FILENAME)
    rnet_part_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_ANNO_FILENAME)
    rnet_neg_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_ANNO_FILENAME)

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)

    imglist_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_TRAIN_IMGLIST_FILENAME)

    chose_count = assemble.assemble_data(imglist_file, anno_list)
    print("RNet train annotation result file path:%s, total num of imgs: %d" % (imglist_file, chose_count))

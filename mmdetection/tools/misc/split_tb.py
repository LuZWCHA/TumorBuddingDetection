# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import numpy as np
from mmengine.fileio import dump, load
from mmengine.utils import mkdir_or_exist, track_parallel_progress

prog_description = '''K-Fold coco split.

To split coco data for semi-supervised object detection:
    python tools/misc/split_coco.py
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='/nasdata2/dataset/research_data/coco_format/v5/')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.',
        default='/nasdata2/dataset/research_data/coco_format/v5/semi_anns/')
    parser.add_argument(
        '--labeled-percent',
        type=float,
        nargs='+',
        help='The percentage of labeled data in the training set.',
        default=[5, 10, 20, 50])
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=1)
    args = parser.parse_args()
    return args


def split_tb(data_root, out_dir, percent, fold):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        out_dir (str): The output directory of coco semi-supervised
            annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """

    def save_anns(name, images, annotations):
        sub_anns = dict()
        sub_anns['images'] = images
        sub_anns['annotations'] = annotations
        # sub_anns['licenses'] = anns['licenses']
        sub_anns['categories'] = anns['categories']
        # sub_anns['info'] = anns['info']

        mkdir_or_exist(out_dir)
        dump(sub_anns, f'{out_dir}/{name}.json')

    # set random seed with the fold
    np.random.seed(fold)
    ann_file = osp.join(data_root, 'train.json')
    anns = load(ann_file)

    image_list = anns['images']
    labeled_total = int(percent / 100. * len(image_list))

    labeled_inds = set(
        np.random.choice(range(len(image_list)), size=labeled_total, replace=False))
    labeled_ids, labeled_images, unlabeled_images = [], [], []

    for i in range(len(image_list)):
        if i in labeled_inds:
            labeled_images.append(image_list[i])
            labeled_ids.append(image_list[i]['id'])
        else:
            unlabeled_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_ids = set(labeled_ids)
    labeled_annotations, unlabeled_annotations = [], []

    for ann in anns['annotations']:
        if ann['image_id'] in labeled_ids:
            labeled_annotations.append(ann)
        else:
            unlabeled_annotations.append(ann)

    # save labeled and unlabeled
    labeled_name = f'train.{fold}@{percent}'
    unlabeled_name = f'train.{fold}@{percent}-unlabeled'

    save_anns(labeled_name, labeled_images, labeled_annotations)
    save_anns(unlabeled_name, unlabeled_images, unlabeled_annotations)


def multi_wrapper(args):
    return split_tb(*args)


if __name__ == '__main__':
    args = parse_args()
    arguments_list = [(args.data_root, args.out_dir, p, f)
                      for f in range(1, args.fold + 1)
                      for p in args.labeled_percent]
    track_parallel_progress(multi_wrapper, arguments_list, args.fold)

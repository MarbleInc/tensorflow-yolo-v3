import argparse
import json
import time
import os
# import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint

from os.path import join
# from common import create_detection_graph, detect, detect_batch

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, required=True, help='Path to evaluation dataset.')
parser.add_argument('--model_dir', type=str, required=True, help='Directory contains model weights and configs.')
parser.add_argument('--gt_class', type=str, default='ground_truth', help='Class corresponds to ground truth.')
parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold to evaluate mertics.')
parser.add_argument('--filter_height', type=int, default=60, help='Height threshold for boxes.')
parser.add_argument('--filter_width', type=int, default=25, help='Width threshold for boxes.')
parser.add_argument('--score_threshold', type=float, default=0.5, help='Score threshold for model detections.')

USE_XLA = False
PRED_SIZE = 448

# def convert_to_original_size(box, original_size):
#     box = box.reshape(2, 2) * original_size
#     return list(box.reshape(-1))

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def get_obj_bbox_vector(obj):
    x = np.array(obj['points']['exterior'])[:, 0]
    y = np.array(obj['points']['exterior'])[:, 1]
    cur_bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
    return cur_bbox


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name.strip()
    return names


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_overlaps(boxes1, boxes2):
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        for j in range(overlaps.shape[0]):
            box1 = boxes1[j]
            overlaps[j, i] = bb_intersection_over_union(box1, box2)
    return overlaps


def compute_matches(gt_boxes, pred_boxes, iou_threshold):
    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    # Loop through predictions and find matching ground truth boxes
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            gt_match[j] = i
            pred_match[i] = j
            break

    return gt_match, pred_match


def bb_filter_det(bbox, filter_size):
    if (bbox[3] - bbox[1]) < filter_size[1] or (bbox[2] - bbox[0]) < filter_size[0]:
        return False
    else:
        return True


def compute_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    elif len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    gt_match, pred_match = compute_matches(gt_boxes, pred_boxes, iou_threshold)

    tp = np.sum(pred_match > -1)
    all_pred = len(pred_match)

    all_gt = len(gt_match)
    return tp, all_pred, all_gt

def load_graph(sess, graph_file):
    gd = tf.GraphDef()
    with tf.gfile.Open(graph_file, 'rb') as f:
        data = f.read()
        gd.ParseFromString(data)
    tf.import_graph_def(gd, name='')


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def process_one_image(pred_boxes, img_path, filter_size, IoU, gt_class):
    ann_basename = os.path.basename(img_path).replace('.png', '.json').replace('.jpg', '.json')
    ann_dirname = join(os.path.dirname(img_path)[:-3], 'ann')
    ann = json.load(open(join(ann_dirname, ann_basename), encoding='utf-8'))
    objects = ann['objects']
    h, w = ann['size']['height'], ann['size']['width']
    gt_boxes = []
    for obj in objects:
        bbox = get_obj_bbox_vector(obj)
        if obj['classTitle'] == gt_class:
            gt_boxes.append(bbox)
        else:
            pass

    # print("gt_boxes: %s" % gt_boxes)
    # print("pred_boxes: %s" % pred_boxes)
    if filter_size is not None:
        gt_boxes = list(filter(lambda x: bb_filter_det(x, filter_size), gt_boxes))
        pred_boxes = list(filter(lambda x: bb_filter_det(x, filter_size), pred_boxes))

    gt_boxes = np.array(gt_boxes)
    pred_boxes = np.array(pred_boxes)

    tp, all_pred, all_gt = compute_precision_recall(gt_boxes, pred_boxes, IoU)
    return tp, all_pred, all_gt

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = []
    for i, image_pred in enumerate(predictions):
        image_result = {}
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in image_result:
                    image_result[cls] = []
                image_result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([bb_intersection_over_union(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
        result.append(image_result)

    return result


def detect_batch(sess, img_paths, tf_input, tf_output, iou_threshold, conf_threshold, gt_class_id):
    input_data = []

    for img_path in img_paths:
        img = Image.open(img_path)
        # input_data.append(np.array(img, dtype=np.float32))
        img_resized = img.resize(size=(PRED_SIZE, PRED_SIZE))
        input_data.append(np.array(img_resized, dtype=np.float32))

    input_data = np.stack(input_data)
    start = time.time()

    detected_boxes = sess.run(tf_output, feed_dict={tf_input: input_data})

    # TODO: look into using tf.image.non_max_suppression instead
    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=conf_threshold,
                                         iou_threshold=iou_threshold)

    bounding_boxes = []
    for image_boxes in filtered_boxes:
        image_bounding_boxes = []
        for cls, bboxs in image_boxes.items():
            if cls != gt_class_id:
                print("filtered out class %s" % cls)
                continue
            for box, score in bboxs:
                box = convert_to_original_size(box, np.array([PRED_SIZE, PRED_SIZE]), np.array(img.size))
                image_bounding_boxes.append(box)
        bounding_boxes.append(image_bounding_boxes)

    end = time.time()
    det_time = end - start
    return bounding_boxes, det_time



def evaluate_on_batches(args):
    # Evaluation parameters
    pr_dir = args.test_dataset  # dir with annotations in supervisely format
    gt_class = args.gt_class  # groundtruth class
    filter_size = (args.filter_width, args.filter_height)
    IoU = args.iou  # Intersection over union threshold to match boxes
    ids_to_classes = load_coco_names(os.path.join(args.model_dir, 'model.names'))
    classes_to_ids = {v: k for k, v in ids_to_classes.items()}
    print(classes_to_ids)
    gt_class_id = classes_to_ids[gt_class]

    img_pathes = []
    for file in os.listdir(pr_dir):
        if not os.path.isdir(join(pr_dir, file)):
            continue
        ds_imgs = os.listdir(join(pr_dir, file, 'img'))
        ds_imgs = [join(pr_dir, file, 'img', x) for x in ds_imgs]
        img_pathes.extend(ds_imgs)

    # Init tf Session
    config = tf.ConfigProto()
    if USE_XLA:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load the frozen graph
    load_graph(sess, os.path.join(args.model_dir, 'model.pb'))

    # Get the input and output tensors
    tf_input = sess.graph.get_tensor_by_name('input:0')
    # print("input tensor:")
    # print(tf_input)
    tf_output = sess.graph.get_tensor_by_name('output:0')
    # print("output tensor:")
    # print(tf_output)

    # detection_graph, session = load_network(args.model_dir)
    _, _ = detect_batch(
        sess, [img_pathes[0], img_pathes[0]],
        tf_input, tf_output, iou_threshold=IoU, conf_threshold=args.score_threshold,
        gt_class_id=gt_class_id)

    tps = 0
    all_preds = 0
    all_gts = 0
    total_det_time = 0

    start_time = time.time()
    for img_batch in batch(img_pathes, 2):
        prediction_res, det_time = detect_batch(
            sess, img_batch, tf_input, tf_output,
            iou_threshold=IoU, conf_threshold=args.score_threshold, gt_class_id=gt_class_id)
        total_det_time += det_time
        for i, image_path in enumerate(img_batch):
            tp, all_pred, all_gt = process_one_image(
                prediction_res[i], image_path, filter_size, IoU, gt_class)
            tps += tp
            all_preds += all_pred
            all_gts += all_gt
        # break  # TODO: debug
    end_time = time.time()

    precision = tps / all_preds if all_preds > 0 else 0
    recall = tps / all_gts if all_gts > 0 else 0
    print('\n\n\n')
    print('######## Evaluation results. ########')
    print('{} images processed'.format(len(img_pathes)))
    print('{} ground truth objects were used to evaluation.'.format(all_gts))
    print('Precision: {} \nRecall: {}'.format(precision, recall))
    print('Average detection time per 2 images: {} s.'.format(total_det_time / np.ceil(len(img_pathes) / 2)))
    # print('Average time(including images uploading and metrics evaluation) per 2 images: {} s.'.format(
    #     (end_time - start_time) / np.ceil(len(img_pathes) / 2)))


if __name__ == "__main__":
    args = parser.parse_args()
    # evaluate_on_single_images(args)
    evaluate_on_batches(args)

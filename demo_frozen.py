# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import glob

from yolo_v3 import non_max_suppression


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image file or dir')
tf.app.flags.DEFINE_string('subset_file', '', 'Optional file containing image names to be selected for inference')
tf.app.flags.DEFINE_string('output_dir', '', 'Output images directory')
tf.app.flags.DEFINE_string('output_file', '', 'Output file containing detections info')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('model', 'yolov3.pb', 'Binary file with detector weights')

tf.app.flags.DEFINE_bool('use_xla', False, 'Runtime optimization')
tf.app.flags.DEFINE_integer('size', 448, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.5, 'IoU threshold')


def load_imgs(src, include_lst_file):
    # Read images from directory (size must be the same) or single input file
    if os.path.isdir(src):
        tmp_img_file_paths = glob.glob(os.path.join(src, '*.jpg'))
        tmp_img_file_paths += glob.glob(os.path.join(src, '*.png'))

        if include_lst_file:
            img_file_paths = []
            imgs_subset = set(line.strip() for line in open(include_lst_file))
            for img_path in tmp_img_file_paths:
                filename = img_path.split('/')[-1]
                img_id = filename.split('.')[0]
                if img_id in imgs_subset or img_id in imgs_subset:
                    img_file_paths.append(img_path)
        else:
            img_file_paths = tmp_img_file_paths
    else:
        img_file_paths = [src]

    # sort the list of files by name
    img_file_paths.sort()

    return img_file_paths


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, img_path):
    draw = ImageDraw.Draw(img)
    dets_str=""

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(img.size))
            # print("%s: %s (%s)" % (cls, score, box))
            # TODO: workaround add +1 to person class to make it compatible with KITTI cls index
            dets_str += "%s %d %.6f %s\n" % (img_path, cls + 1, score, ' '.join(map(str,map(int, box))))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)

    return dets_str


def convert_to_original_size(box, original_size):
    box = box.reshape(2, 2) * original_size
    return list(box.reshape(-1))

def load_graph(sess, graph_file):
    gd = tf.GraphDef()
    with tf.gfile.Open(graph_file, 'rb') as f:
        data = f.read()
        gd.ParseFromString(data)
    tf.import_graph_def(gd, name='')

def main(argv=None):
    classes = load_coco_names(FLAGS.class_names)

    # Init tf Session
    config = tf.ConfigProto()
    if FLAGS.use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load the frozen graph
    load_graph(sess, FLAGS.model)

    # Get the input and output tensors
    tf_input = sess.graph.get_tensor_by_name('input:0')
    tf_output = sess.graph.get_tensor_by_name('output:0')

    # load image path(s) from disk
    img_paths = load_imgs(FLAGS.input_img, FLAGS.subset_file)
    det_boxes = ""
    for img_path in tqdm(img_paths):
        # load image in memory
        img = Image.open(img_path)

        # create input batch
        input_data = np.stack([np.array(img, dtype=np.float32)])

        # infer bounding boxes
        detected_boxes = sess.run(tf_output, feed_dict={tf_input: input_data})

        # TODO: look into using tf.image.non_max_suppression instead
        # merge boxes using nms
        filtered_boxes = non_max_suppression(
            detected_boxes, confidence_threshold=FLAGS.conf_threshold, iou_threshold=FLAGS.iou_threshold)

        # add detected bbox to string and render them on a given image
        det_boxes += draw_boxes(filtered_boxes, img, classes, img_path)

        # save image with prediction boxes to disk if output dir specified
        if FLAGS.output_dir:
            img.save(os.path.join(FLAGS.output_dir, img_path.split('/')[-1]))

    # write detections to a file
    with open(FLAGS.output_file, 'w') as out:
        out.write(det_boxes)


if __name__ == '__main__':
    tf.app.run()

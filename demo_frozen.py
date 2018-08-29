# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from yolo_v3 import non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('model', 'yolov3.pb', 'Binary file with detector weights')

tf.app.flags.DEFINE_bool('use_xla', False, 'Runtime optimization')
tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            print("%s: %s (%s)" % (cls, score, box))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def load_graph(sess, graph_file):
    gd = tf.GraphDef()
    with tf.gfile.Open(graph_file, 'rb') as f:
        data = f.read()
        gd.ParseFromString(data)
    tf.import_graph_def(gd, name='')

def main(argv=None):
    img = Image.open(FLAGS.input_img)

    # FIXME: do the resize in tensorflow so I can pass any size input image
    img_resized = img.resize(size=(FLAGS.size, FLAGS.size))
    input_data = np.stack([np.array(img_resized, dtype=np.float32)])
    print("input shape: %s" % (input_data.shape,))

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    # inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

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
    print("input tensor:")
    print(tf_input)
    tf_output = sess.graph.get_tensor_by_name('output:0')
    print("output tensor:")
    print(tf_output)

    detected_boxes = sess.run(tf_output, feed_dict={tf_input: input_data})

    # TODO: look into using tf.image.non_max_suppression instead
    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()

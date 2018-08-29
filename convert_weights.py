#!/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('output_file', '', 'Protobuf file with frozen weights and graph')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def main(argv=None):
    # placeholder for detector inputs
    classes = load_coco_names(FLAGS.class_names)
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], name="input")

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NCHW')
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    boxes = detections_boxes(detections)
    boxes = tf.identity(boxes, "output")

    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(load_ops)

        print("Finished loading graph")
        print("%d ops in the base graph." % len(sess.graph.get_operations()))

        # save the (trained) model
        # We use a built-in TF helper to export variables to constants
        output_node_names = 'output'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        tf.train.write_graph(output_graph_def, '', FLAGS.output_file, False)
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    tf.app.run()

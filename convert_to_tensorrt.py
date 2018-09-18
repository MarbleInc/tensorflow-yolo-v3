import os.path
import tensorflow as tf
import tensorrt as trt
from tensorflow.python.platform import gfile
import uff
from tensorrt.parsers import uffparser

import numpy as np

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_file', 'yolov3.pb', 'Protobuf file with frozen weights and graph')
tf.app.flags.DEFINE_string('output_file', '', 'Output graph with tensorRT')
tf.app.flags.DEFINE_string('precision', 'FP32', 'Precision: FP32, FP16, int8, native')
tf.app.flags.DEFINE_integer('workspace_size', 2000, 'workspace size in MB')
tf.app.flags.DEFINE_integer('max_batch_size', 2, 'Maximum batch size')


def get_frozen_graph(modelpath):
    with gfile.FastGFile(modelpath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def
#
# def save_tensorrt_graph(frozen_graph_def, outfile):
#     output_node_name = 'output'
#     batch_size = 2
#     precision = 'FP32'
#     workspace_size = 2000
#     trt_graph = trt.create_inference_graph(
#         input_graph_def=frozen_graph_def,
#         outputs=output_node_name,
#         max_batch_size=batch_size,
#         max_workspace_size_bytes=workspace_size,
#         precision_mode=precision,
#         minimum_segment_size=3)

def convert_from_frozen_graph(modelpath):
    # tf_model = get_frozen_graph(modelpath)
    uff_model = uff.from_tensorflow_frozen_model(modelpath, ["output"])
    parser = uffparser.create_uff_parser()
    parser.register_input("input", (None, None, None, 3), 0)
    parser.register_output("output")

    engine = trt.utils.uff_to_trt_engine(
        G_LOGGER,
        uff_model,
        parser,
        FLAGS.max_batch_size,
        FLAGS.workspace_size)

if __name__ == "__main__":
    convert_from_frozen_graph(FLAGS.input_file)

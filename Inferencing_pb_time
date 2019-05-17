#Code to inference from a pb file and get total run time.
#FP16 takes lesser time as compared to FP32

#Code-

import tensorflow as tf
import argparse
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import timeit

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    #parser.add_argument("--frozen_model_filename", default='/home/nvidia/Desktop/alexnet/alexnet/model.pb', type=str, help="Frozen model file to import")
    parser.add_argument("--frozen_model_filename", default='/home/nvidia/Desktop/ojas/flow_RT_frozen_tensorrt.pb', type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    #x = graph.get_tensor_by_name('prefix/conv_1_input:0')
    #y = graph.get_tensor_by_name('prefix/output/Sigmoid:0')
    x = graph.get_tensor_by_name('prefix/conv2d_1_input:0')
    y = graph.get_tensor_by_name('prefix/activation_9/Sigmoid:0')
    img = cv2.imread('/home/nvidia/Desktop/alexnet/alexnet/0000000052.png')
    img2 = cv2.resize(img,(224,224))
    img2 = np.reshape(img2,(-1,224,224,3))
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        start = timeit.default_timer()
        y_out = sess.run(y, feed_dict={x: img2})
        stop = timeit.default_timer()
        print(y_out) 
        print("time taken: ",stop - start)

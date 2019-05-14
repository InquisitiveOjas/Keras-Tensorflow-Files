# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:58:31 2019

@author: ojasr
"""
# Code 1 -> Generates error due to "config"

import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = 'Downloads/Alexnet_May_7.pb' #path to your .pb file

with tf.Session(config=config) as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    
wts = [n for n in graph_nodes if n.op=='Const']

from tensorflow.python.framework import tensor_util

for n in wts:
    print ("Name of the node - " ,n.name)
    print ("Value - ") 
    print (tensor_util.MakeNdarray(n.attr['value'].tensor))

# Code 2 works commenting "pbfile = sys.argv[1]"

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:51:59 2019

@author: ojasr
"""

import tensorflow as tf
import sys


## In tensorflow the weights are also stored in constants ops
## So to get the values of the weights, you need to run the constant ops
## It's a little bit anti-intution, but that's the way they do it

#construct a GraphDef

#pbfile = sys.argv[1] -> Discuss the use of this command
graph_def = tf.GraphDef()
with open('Downloads/Alexnet_May_7.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

#import the GraphDef to the global default Graph
tf.import_graph_def(graph_def, name='')


# extract all the constant ops from the Graph
# and run all the constant ops to get the values (weights) of the constant ops
constant_values = {}
with tf.Session() as sess:
    constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
    for constant_op in constant_ops:
        value =  sess.run(constant_op.outputs[0])
        constant_values[constant_op.name] = value

        #In most cases, the type of the value is a numpy.ndarray.
        #So, if you just print it, sometimes many of the values of the array will
        #be replaced by ...
        #But at least you get an array to python object, 
        #you can do what other you want to save it to the format you want

        print (constant_op.name, value)
        
        
# Extracting the architecture / model summary

# https://github.com/tensorflow/tensorflow/issues/8854
        
import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='Downloads/Alexnet_May_7.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='OneDrive\Desktop\LogDir' #Enter Location
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

# To view the architecture in the form of a graph type in anaconda prompt: tensorboard --logdir=C:\Users\ojasr\OneDrive\Desktop\LogDir --host localhost --port 6006



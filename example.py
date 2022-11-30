#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import sonnet as snt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf


SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)



# # Provide your own functions to generate graph-structured data.
# input_graphs = get_graphs()

# # Create the graph network.
# graph_net_module = gn.modules.GraphNetwork(
#     edge_model_fn=lambda: snt.nets.MLP([32, 32]),
#     node_model_fn=lambda: snt.nets.MLP([32, 32]),
#     global_model_fn=lambda: snt.nets.MLP([32, 32]))

# # Pass the input graphs to the graph network, and return the output graphs.
# output_graphs = graph_net_module(input_graphs)
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import gensim
import networkx as nx
from node2vec import Node2Vec

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

from networkx import DiGraph

#Dataframe

interactome = pd.read_csv("bio-decagon-ppi.csv")
print(interactome)


#Defining log object to notify the steps updates

class AbstractSimpleLog():
    def log(self, msg):
        raise Exception("log method must be implemented")

class PrintLog(AbstractSimpleLog):
    def log(self, msg):
        print(msg)
        


#Default name for embeddign file
EMBEDDING_FILE = "embeddings.txt"

#Log object. To long computations

logger = PrintLog()


#Graph embedding

logger.log("Loading graph from {}".format(interactome))
graph = nx.read_edgelist(interactome, delimiter=" ", create_using=DiGraph())
logger.log("Graph loaded")

logger.log("Computing transition probabilities")
n2v = Node2Vec(graph, dimensions= 715612, walk_length=80, num_walks=50, workers=4, p=1, q=1)
logger.log("Transitions probabilities computed")

logger.log("Starting Node2Vec embedding")
n2v_model = n2v.fit(window=80, min_count=1, batch_words=64)
logger.log("Node2Vec embedding created")

logger.log("Saving embedding file")
n2v_model.wv.save_word2vec_format(EMBEDDING_FILE)
logger.log("Embedding file saved")

#Data processing
import numpy as np
from numpy import array

vectors = np.loadtxt('embeddings.txt',dtype='str', delimiter=' ',skiprows=1)
print(vectors)
def to_embedded(n):
    return vectors[n,:]
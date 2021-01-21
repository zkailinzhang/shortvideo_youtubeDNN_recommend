
import os 
import copy 
import numpy as np 
import tensorflow as tf 
import faiss
from tensorflow.python.saved_model import tag_constants

PATH = os.path.dirname(os.path.abspath(__file__))






def prepare_base_data(data,):

    return data


def online_data(data):
    
     
    return data                      


def  load_model(feed ,saved_dir ):


    saved_model_dir = saved_dir
    #signature_key = 'test_signature'
    sign = 'serving'
    output_key = 'outputs'
    
    signature_key = tf.saved_model.tag_constants.SERVING

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        meta_graph_def = tf.saved_model.loader.load(sess, [signature_key], saved_model_dir)
        signature = meta_graph_def.signature_def

        y_tensor_name = signature[sign].outputs[output_key].name

        y = sess.graph.get_tensor_by_name(y_tensor_name)
      
        inp ={}
        for k, v in feed.items():
            tensor_name = signature[sign].inputs[k].name
            name = sess.graph.get_tensor_by_name(tensor_name)
            inp[name]=v 


        proc = sess.run(y, feed_dict=inp)

    return proc


def annSearch(uvec,ivec,topk):
    emds=64
    
    indextree = faiss.IndexFlatIP(emds)

    indextree.add(ivec)

    Dis,Index= indextree.search(np.array(uvec).astype('float32'),topk) 
    return Dis,Index

if __name__ == "__main__":

    test = {"logid":"requestId","user_id_ph":40252,"his_item_seq_ph":[1971,363,565],"cathis_cate_seq_phegory":[807,657,657],\
    "his_tag_seq_ph":[1579,657,940],"his_len_ph": 3}

    dataonline = [{"user_id_ph":40252,"his_item_seq_ph":[1971,363,565],"his_cate_seq_ph":[807,657,657],"his_tag_seq_ph":[1579,657,940],"his_len_ph": 3},
    {"user_id_ph":40252,"his_item_seq_ph":[1971,363,565],"his_cate_seq_ph":[807,657,657],"his_tag_seq_ph":[1579,657,940],"his_len_ph": 3}]
    dataonline = {"user_id_ph":[40252,40252,],"his_item_seq_ph":[[1971,363,565],[1971,363,565]],"his_cate_seq_ph":[[807,657,657],[807,657,657]],"his_tag_seq_ph":[[1579,657,940],[1579,657,940]],"his_len_ph": [3,3]}

    model_path = os.path.join(PATH,'serving/1/')
    uvec = load_model(dataonline,model_path)
    print(uvec)
    ivec = np.load('./weights.npy')
    Dis,Index = annSearch(uvec,ivec,10)
    print(Dis,Index)
# [[0.00100544 0.00084022 0.00083886 0.00083189 0.00082821 0.00082384
#   0.00082008 0.0008171  0.00081407 0.00080773]
#  [0.00100544 0.00084022 0.00083886 0.00083189 0.00082821 0.00082384
#   0.00082008 0.0008171  0.00081407 0.00080773]] [[39149 28977   950 41537 12481 21623 32898 17851 35884 12272]
#  [39149 28977   950 41537 12481 21623 32898 17851 35884 12272]]


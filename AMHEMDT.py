import math
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import tqdm
from numpy import random
import pandas as pd

from utils import *
import random
import math
from sklearn.preprocessing import Normalizer

from sklearn.ensemble import RandomForestClassifier

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/miRNA',
                        help='Input dataset path')
    
    parser.add_argument('--features', type=str, default='./data/miRNA/feature_128_1.txt',
                        help='Input node features')

    parser.add_argument('--epoch', type=int, default=1,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=32,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=32,
                        help='Number of edge embedding dimensions. Default is 10.')
    
    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--negative-samples', type=int, default=20,
                        help='Negative samples for optimization. Default is 5.')
    
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    
    return parser.parse_args()

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32)) 

def train_model(network_data,test_data_by_type_embedding,test_positive_classifier,train_positive_classifier,file_name, feature_dic, log_name):
    
   
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)
    vocab, index2word = generate_vocab(all_walks)

    
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)
    

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples 

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    graph = tf.Graph()

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic is not None:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))
            linear = tf.layers.Dense(units=embedding_size, activation=tf.nn.tanh, use_bias=True)

            embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples])
        
        # Look up embeddings for nodes
        if feature_dic is not None:
            node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = tf.matmul(node_embed, embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        
        if feature_dic is not None:
            node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
            node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]), [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
        else:
            node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
            node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        
        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])

        if feature_dic is not None:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, feature_weights)

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        g_iter = 0
        best_score = 0
        patience = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm.tqdm(batches,
                                desc="epoch %d" % (epoch),
                                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            
            
            
        for i in range(edge_type_count):
            for j in range(num_nodes):
                final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
           
        print(final_model.keys())
        print(edge_types)

        node_miRNA_Nembedding=[]
        node_disease_Nembedding=[]
        type_edge_Nembedding=[]
            
        test_data_by_type_classifier = load_training_data(train_positive_classifier[:,:3])  
        
       
            
            
            
            
        test_negative_classifier_temp= np.ones((int(test_positive_classifier.shape[0]),3),int)

        a_test=0
        b_test=len(np.where(test_positive_classifier[:,0] == int(edge_types[0]))[0])
        for i in range(edge_type_count):
            if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    
                train_p_num = len(np.where(train_positive_classifier[:,0] == int(edge_types[i]))[0])
                test_p_num = len(np.where(test_positive_classifier[:,0] == int(edge_types[i]))[0])
                    
                print('test_positive_num',test_p_num)
                    
                All_negative_classifier_test,node_miRNA,node_disease,type_edge = Get_score_Nembedding(train_p_num,test_p_num,final_model[edge_types[i]], test_data_by_type_classifier[edge_types[i]],test_data_by_type_embedding[edge_types[i]],edge_types[i])
                    
                   
                node_miRNA_Nembedding.extend(node_miRNA)
                node_disease_Nembedding.extend(node_disease)
                type_edge_Nembedding.extend(type_edge)

                    
                    
                    
                    
                test_negative_classifier_temp[a_test:b_test,:]= All_negative_classifier_test
                     
                    
                  
                a_test+=test_p_num
                print('i',i)
                if i<4:
                        
                    b_test +=len(np.where(test_positive_classifier[:,0] ==int(edge_types[i+1]))[0])
                    
                   
        train_negative_classifier_temp= np.array([type_edge_Nembedding,node_miRNA_Nembedding,node_disease_Nembedding ]).T        

        print('train_negative_classifier_temp',train_negative_classifier_temp.shape)
        print('test_negative_classifier_temp',test_negative_classifier_temp.shape)
            
            
        test_negative_classifier= np.column_stack((test_negative_classifier_temp,np.zeros((test_negative_classifier_temp.shape[0]), int)))
        train_negative_classifier= np.column_stack((train_negative_classifier_temp,np.zeros((train_negative_classifier_temp.shape[0]), int)))
            
        print('negative',test_negative_classifier.shape)
        print('negative',train_negative_classifier.shape)
            
        test_classifier= np.row_stack((test_positive_classifier,test_negative_classifier))
        train_classifier= np.row_stack((train_positive_classifier,train_negative_classifier))
            
        print('negative',test_classifier.shape)
        print('negative',train_classifier.shape)
            
        true_edge_test_by_type, false_edge_test_by_type = load_testing_data(test_classifier)
        true_edge_train_by_type, false_edge_train_by_type = load_testing_data(train_classifier)
            
            ####以下暂未修改 
        test_feature=[]
        test_lable=[]
        train_feature=[]
        train_lable=[]
        nonembedding_lable=[]
        nonembedding_prob=[]
            
       
            
        for j in range(edge_type_count):
            if args.eval_type == 'all' or edge_types[j] in args.eval_type.split(','):
                print('edge_types[j]',edge_types[j])
                feature_test, lable_test,nonembedding_test_lable,nonembedding_test_prob = select_feature(final_model[edge_types[j]], true_edge_test_by_type[edge_types[j]], false_edge_test_by_type[edge_types[j]])
                test_feature.extend(feature_test)
                test_lable.extend(lable_test)
                nonembedding_lable.extend(nonembedding_test_lable)
                nonembedding_prob.extend(nonembedding_test_prob)
                    
                feature_train, lable_train,nonembedding_lable_train,nonembedding_prob_train = select_feature(final_model[edge_types[j]], true_edge_train_by_type[edge_types[j]], false_edge_train_by_type[edge_types[j]])
                train_feature.extend(feature_train)
                train_lable.extend(lable_train)
                    
                print('ending')
                    
           
            
        test_feature_matrix = np.array(test_feature)
        test_label_vector = np.array(test_lable)
        train_feature_matrix = np.array(train_feature)
        train_label_vector = np.array(train_lable)
           
       
            
        normalizer = Normalizer().fit(train_feature_matrix)
        train_feature_matrix_N=normalizer.transform(train_feature_matrix)
        test_feature_matrix_N=normalizer.transform(test_feature_matrix)    

        clf = RandomForestClassifier(random_state=1, n_estimators=350, oob_score=False, n_jobs=-1)
        clf.fit(train_feature_matrix_N, train_label_vector)#模型训练的时候没有负样本
        predict_y_proba = clf.predict_proba(test_feature_matrix_N)[:, 1]
            
          
             
            
            ##将孤立点的关联概率预测为0
        print('before',len(list(test_lable)),len(list(predict_y_proba)))
        test_label_vector = np.array(list(test_lable) + nonembedding_lable)
        predict_y_proba = np.array(list(predict_y_proba) + nonembedding_prob)
            
       
            
            
        AUPR,AUC,f1=model_evaluate(test_label_vector, predict_y_proba)
        #AUPR,AUC,f1= get_metrics_TDRC(test_label_vector, predict_y_proba)
        print(AUPR,AUC,f1)
          
            
            

    return AUPR,AUC,f1

def Experiments(args,mir_dis_data,seed):
    k_folds = 10
    seed= random.randint(1,50)
    index_matrix = np.array(np.where(mir_dis_data == 1)).T
    index_matrix[:,1]=index_matrix[:,1]+695
   
    index_matrix_negative = np.array(np.where(mir_dis_data== 0)).T
    index_matrix_negative[:,1]=index_matrix_negative[:,1]+695
   
    
    np.random.seed(seed)
    positive_randomlist = [i for i in range(index_matrix.shape[0])]
    random.shuffle(positive_randomlist)
    
   

    metrics_result = np.zeros((1, 3))
        # metrics_CP = np.zeros((1, 7))
    print("seed=%d, evaluating miRNA-disease...." % (seed))

    for k in range(k_folds):#k=0,1,2,3,4
        print("------this is %dth cross validation------"%(k+1))
        if k != k_folds-1:#取一份正样本作为测试集一部分
            positive_test = positive_randomlist[k*int(index_matrix.shape[0]/k_folds):(k+1)*int(index_matrix.shape[0]/k_folds)]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))
        else:
            positive_test = positive_randomlist[k * int(index_matrix.shape[0] / k_folds)::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))
        
        ##嵌入train positive样本
        train_sample_temp=index_matrix[positive_train,:]
        train_positive_embedding=np.column_stack((train_sample_temp[:,2],train_sample_temp[:,0:2]))
        
        ##嵌入train negative样本
        train_negative_embedding=np.column_stack((index_matrix_negative[:,2],index_matrix_negative[:,0:2]))
        
        
        ##分类器test样本，只要正样本，负样本根据embedding 相似性再选择  
        test_sample_positive=index_matrix[positive_test,:]
        test_lable_vector_classifier= np.ones((test_sample_positive.shape[0]), int)
        test_positive_classifier=np.column_stack((test_sample_positive[:,2],test_sample_positive[:,0:2],test_lable_vector_classifier))
        
        
        ##分类器train样本，只要正样本，负样本根据embedding 相似性再选择
        train_lable_vector_classifier = np.ones((train_sample_temp.shape[0]), int)
        train_positive_classifier = np.column_stack((train_sample_temp[:,2],train_sample_temp[:,0:2],train_lable_vector_classifier))
        
        
       
        
        
        
        args = parse_args()
        file_name = args.input
        print(args)
        if args.features is not None:
            feature_dic = {}
            with open(args.features, 'r') as f:
                first = True
                for line in f:
                    if first:
                        first = False
                        continue
                    items = line.strip().split('\t')
                    feature_dic[items[0]] = items[1:]
        else:
            feature_dic = None
        
       
        pd.DataFrame(feature_dic).to_csv(r'feature_dic.csv')
        log_name = file_name.split('/')[-1] + f'_evaltype_{args.eval_type}_b_{args.batch_size}_e_{args.epoch}'

       
        training_data_by_type_embedding = load_training_data(train_positive_embedding)
        test_data_by_type_embedding = load_training_data(train_negative_embedding)
    
        
        average_auc, average_f1, average_pr = train_model(training_data_by_type_embedding,test_data_by_type_embedding,test_positive_classifier,train_positive_classifier,file_name, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())))

        metrics_result+=np.array([average_auc, average_f1, average_pr])
        name = 'Result/miRNA_disease_seed=' + str(k) + '.csv'
        np.savetxt(name, metrics_result, delimiter=',')



        
    result = np.around(metrics_result / 10, decimals=4)
    return result



    


   
if __name__ == "__main__":

    miRNA_sim_matrix = np.loadtxt('./miRNA_sim_matrix.csv', delimiter=',', dtype=float)
    dis_sim_matrix = np.loadtxt('./disease_sim_matrix.csv',delimiter=',', dtype=float)
    
    miRNA_sim_matrix_emb=Node_vec(miRNA_sim_matrix)  
    dis_sim_matrix_emb=Node_vec(dis_sim_matrix)
    
    root=r'./data/type_data'
    a0= np.loadtxt('./data/type_data/Type1.csv',delimiter=',', dtype=np.int32)
    a1= np.loadtxt('./data/type_data/Type2.csv',delimiter=',', dtype=np.int32)
    a2= np.loadtxt('./data/type_data/Type3.csv',delimiter=',', dtype=np.int32)
    a3= np.loadtxt('./data/type_data/Type4.csv',delimiter=',', dtype=np.int32)
    a4= np.loadtxt('./data/type_data/Type5.csv',delimiter=',', dtype=np.int32)
    
    mir_dis_data=np.stack([a0,a1,a2,a3,a4],axis=2)
    args = parse_args()
    experiment=Experiments(args,mir_dis_data,2)
    print(experiment)
    

   

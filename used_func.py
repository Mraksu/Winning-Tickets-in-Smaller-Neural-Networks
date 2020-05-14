''' This file contains some of the most used funtcions in the
main experiment'''

import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import json
import os

def parameter_count(model_name,model,verbose=0):
    '''To check the number of the nonzero parameters in each layer,
    also to control the total number of parameter
    if verbose=1 than it will print out the number of the
    nonzero parameters in the model.
    '''
    if verbose==0:

        sum_params = 0

        for layer in model.trainable_weights:
            sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

        return sum_params
    else:
        print("\nIn {} Number of nonzero parameters in each layer are: \n".format(model_name))

        sum_params = 0

        for layer in model.trainable_weights:
            print(tf.math.count_nonzero(layer, axis = None).numpy())
            sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

        print("\nTotal number of trainable parameters = {0}\n".format(sum_params))

        return sum_params

def define_pruning_params(target=0.0,begin=0,end=0,freq=100):

    '''This function is used to create pruning parameters that is used in the
    pruned_nn model.

    check the official website for the information about the arguments
    https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity

    '''

    if target==0.0:
        prun={
        'pruning_schedule': sparsity.ConstantSparsity(
            target_sparsity=target, begin_step=begin,
            end_step = end, frequency=freq
        )
        }
        return [prun,prun]
    else:
        prun={
        'pruning_schedule': sparsity.ConstantSparsity(
            target_sparsity=target, begin_step=begin,
            end_step = end, frequency=freq
        )
        }
        #according to the paper we should use half of the pruning rate  for the layers that are connected to
        #output layer
        prun_out={
        'pruning_schedule': sparsity.ConstantSparsity(
            target_sparsity=target/2, begin_step=begin,
            end_step = end, frequency=freq
        )
        }
        return [prun,prun_out]

def pruning_rounds(model,pruning_percentage=0.2,target=0.01):
    '''This function calculates the number of pruning pruning_rounds
    and corresponding pruning rates for each rounds
    Arguments:
    pruning_percentage= Desired pruning percentage for iterative pruning rounds
    target= Target sparsity value

    returns number of pruning rounds and corresponding pruning rates
     '''
    dense1=tf.math.count_nonzero(model.trainable_weights[0],axis=None).numpy()
    dense2=tf.math.count_nonzero(model.trainable_weights[2],axis=None).numpy()
    op=tf.math.count_nonzero(model.trainable_weights[4],axis=None).numpy()

    total_param=dense1+dense2+op

    print('Number of initial parameters:')
    print('dense1:{}, dense2:{}, op:{}, total:{}'.format(dense1,dense2,op,total_param))

    dense_p=pruning_percentage
    op_p=dense_p/2
    min_param=total_param*target #here I will go as far as %1 percent of total weights since in the original paper results
    #are not reliable after 3.6%
    print('min possible params:{}'.format(min_param))

    initial_param=total_param.copy()
    num_prun_rounds=0
    while total_param>min_param:
        dense1=dense1*(1-dense_p)
        dense2=dense2*(1-dense_p)
        op=op*(1-op_p)
        total_param=dense1+dense2+op
        num_prun_rounds+=1
        print('round:{}, %weights remaining:{:.2f}'.format(num_prun_rounds,1-(initial_param-total_param)/initial_param))
        print('dense1:{:.0f}, dense2:{:.0f}, op:{:.0f}, total:{:.0f}'.format(dense1,dense2,op,total_param))

    print('total number of pruning rounds:{}'.format(num_prun_rounds))

    #lets save the pruning rates for every epoch that we are going to use

    prun_rates=[]
    rate=100
    for i in range(num_prun_rounds):
        rate-=rate*dense_p
        prun_rates.append((100-rate)/100)
    print(prun_rates)
    return prun_rates,num_prun_rounds

def prune_network(model,pruning_percentage=0.2):
    '''This function prunes the lowest weights in each layer in the network with given percentage.
    The layer connected to the output is pruned half of the rate.

    Arguments:

    model: The Keras model object that is going to be pruned.
    pruning_percentage: The pruning percentage.

    modifies and prunes the model
    '''


    l=0
    for layer in model.layers:
        if l<2:
            orig_shape=layer.get_weights()[0].shape
            flat_weights=layer.get_weights()[0].flatten()
            sorted_weights=np.sort(np.abs(flat_weights))
            cutoff_index=np.floor(pruning_percentage*sorted_weights.size).astype(np.int)
            cutoff=sorted_weights[cutoff_index]
            flat_weights=np.where(np.abs(flat_weights)<=cutoff,0,flat_weights)
            new_weights=[1,2] #to assign new weights
            new_weights[0]=np.reshape(flat_weights,orig_shape)
            new_weights[1]=np.zeros((orig_shape[1],))
            layer.set_weights(new_weights)
            l+=1
        else:
            orig_shape=layer.get_weights()[0].shape
            flat_weights=layer.get_weights()[0].flatten()
            sorted_weights=np.sort(np.abs(flat_weights))
            cutoff_index=np.floor((pruning_percentage/2)*sorted_weights.size).astype(np.int)
            cutoff=sorted_weights[cutoff_index]
            flat_weights=np.where(np.abs(flat_weights)<=cutoff,0,flat_weights)
            new_weights=[1,2] #to assign new weights
            new_weights[0]=np.reshape(flat_weights,orig_shape)
            new_weights[1]=np.zeros((orig_shape[1],))
            layer.set_weights(new_weights)

def encode_save_json(dic,filename):
    '''Some data types are not supported by json.
    this functions encodes the given dictionary and
    saves it as a json file'''
    for i in dic.keys():
        for k,v in dic[i].items():
            if type(v)==np.ndarray:
                dic[i][k]=v.tolist()
            else:
                continue

    with open(filename, 'w') as fp:
        json.dump(dic, fp)
    if os.path.exists(filename):
        print('Saved successfuly')
    else:
        print('file does not saved, something wrong')

def decode_json(file_name):
    '''This function decodes the json into dictionary format'''
    with open(file_name, 'r') as fp:
        json_st=fp.read()
        dic=json.loads(json_st)
    new_dic=dict()
    for i in dic.keys():
        new_dic[int(i)]=dic[i]
    for i in new_dic.keys():
        for k,v in new_dic[i].items():
            if type(v)==list:
                new_dic[i][k]=np.array(v)
            else:
                continue
    return new_dic

import tensorflow as tf

def parameter_count(model_name,model):
    '''To check the number of parameters in each layer,
    also to control the total number of parameter'''
    
    print("\nIn {} Number of nonzero parameters in each layer are: \n".format(model_name))

    sum_params = 0

    for layer in model.trainable_weights:
        print(tf.math.count_nonzero(layer, axis = None).numpy())
        sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

    print("\nTotal number of trainable parameters = {0}\n".format(sum_params))

    return sum_params

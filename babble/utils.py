'''
Created on Aug 11, 2021

@author: Navid Dianati
'''

import numpy as np


def get_dataset_from_array(data, batch_size, sequence_length, sequence_stride):
    """
    Batchify and generate training data pairs:
    Given a series of input data points, form a single batch of data
    in an RNN-friendly format: if data is 2D, rows of data are time steps 
    and columns are features. We form a 3D array X of shape 
    (batch_size, sequence_length, data.shape[1]) such that X[i, :, :]
    is a contiguous subarray of data, offset in data from X[i+1, :, :] by
    sequence_stride. if data is 1D, the X is 2-dimentional where each 
    row is a contiguous subarray of data and each element is offest from
    the one in a row below/above by sequence_stride.
    """
    rank = len(data.shape)
    N = data.shape[0]

    # Make sure there are enough timesteps in the data
    # for an array of shape (batch_size, sequence_length)
    if N < batch_size * sequence_length:
        raise ValueError('Not enough rows in data')

    # indices of the starting points of the sequences
    # in the batch.
    indices = np.arange(0, batch_size) * sequence_stride
    
    if rank == 2:
        n_features = data.shape[1]

        # Allocate memory for 3d array
        X = np.zeros((batch_size, sequence_length, n_features), dtype=data.dtype)

        for i, start in enumerate(indices):
            X[i,:,:] = data[start:start + sequence_length,:]
    
    if rank == 1:
        N = data.shape[0]

        # Allocate memory for 2d array
        X = np.zeros((batch_size, sequence_length), dtype=data.dtype)
        
        for i, start in enumerate(indices):
            X[i,:] = data[start:start + sequence_length]
    
    return X
    
    
def data_dispatcher(text, text_onehot, batch_size=16, sequence_length=8, max_batches=1e6, verbose=False):
    """ Given a list of cleaned up and finalized token indices,
    Divide the "text" into <batch_size> contiguous sub_lists,"""
    
    # The input data will consist of that many rows and in successive batches
    # that row will continue the sequence corresponding to that block

    assert text.shape[0] == text_onehot.shape[0]
    
    # Length of the text: number of tokens
    text_length = len(text)

    # Truncate sequence_stride to a whole multiple
    # of sequence_length and truncate text to a whole 
    # multiple of sequence_stride
    sequence_stride = text_length // batch_size
    sequence_stride -= sequence_stride % sequence_length
    text_length = sequence_stride * batch_size
    text = text[:text_length]
    
    if verbose:
        print("text_length: {}".format(text_length))
        print("sequence_length: {}".format(sequence_length))
        print("sequence_stride: {}".format(sequence_stride))

    batch_counter = 0
    while batch_counter < max_batches:
        for i in range(0, sequence_stride, sequence_length):
            X = get_dataset_from_array(
                    text[i:],
                    sequence_length=sequence_length,
                    sequence_stride=sequence_stride,
                    batch_size=batch_size
                )

            y = get_dataset_from_array(
                    text_onehot[i + 1:],
                    sequence_length=sequence_length,
                    sequence_stride=sequence_stride,
                    batch_size=batch_size
                )

            yield X, y
            batch_counter += 1

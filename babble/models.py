'''
Created on Aug 11, 2021

@author: Navid Dianati
'''

import tensorflow.keras as keras


def get_model_1(input_dim, output_dim, batch_size, sequence_length):
        """Model expects a 2D input array where the value at
        each timestep is integer-encoded"""
        
        # LSTM latent dimension
        latent_dim = 1000
        
        layer1 = keras.layers.Embedding(input_dim=input_dim, output_dim=1000, batch_input_shape=(batch_size, sequence_length))
        layer2 = keras.layers.LSTM(latent_dim, return_sequences=True, stateful=True,)
        layer3 = keras.layers.BatchNormalization()
        layer4 = keras.layers.Dropout(0.2)
        
        layer5 = keras.layers.Dense(1000, activation="relu")
        layer6 = keras.layers.BatchNormalization()
        layer7 = keras.layers.Dropout(0.2)
        
        layer8 = keras.layers.Dense(1000, activation="relu")
        layer9 = keras.layers.BatchNormalization()
        layer10 = keras.layers.Dropout(0.2)
        
        layer11 = keras.layers.Dense(output_dim, activation="softmax")
        
        model = keras.Sequential(
            [
                layer1,
                layer2,
                layer3,
                layer4,
                layer5,
                layer6,
                layer7,
                layer8,
                layer9,
                layer10,
                layer11,
            ]
        )
        return model

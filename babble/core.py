
'''
Created on Aug 11, 2021

@author: Navid Dianati
'''
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from . import models, utils, mycallbacks
import tensorflow.keras as keras


class BabbleBot:

    def __init__(self):
        
        # Integer and one-hot token encoders 
        self.token_indexer = LabelEncoder()
        self.token_one_hot_encoder = OneHotEncoder(sparse=False)
        
        self.text_tokenized = []
        self.set_tokens = set()
        self.text_integer_encoded = None
        self.text_one_hot_encoded = None
        
        self.batch_size = 32
        self.sequence_length = 16
        
    def truncate_tokens(self, list_tokens, null_token="<UNKN>"):
        """Perform any desired substitutions, e.g., replace
        all numbers with a <NUM> token. Override if necessary."""
        
        # Drop tokens with freq <= 2
        counts = pd.Series(list_tokens).value_counts()
        rare_tokens = set(list(counts[counts <= 2].index))
        list_tokens = [null_token if t in rare_tokens else t for t in list_tokens]
        return list_tokens

    def parse_text(self, text):
        # Apply any "truncation", e.g., replace rare
        # Tokens with "<UNKN>"
        self.text_tokenized = self.truncate_tokens(text)
        
        # Set of all tokens
        self.set_tokens = set(self.text_tokenized)
        print("Total number tokens, and number of unique tokens: {:,}, {:,}".format(
            len(self.text_tokenized), len(self.set_tokens))
        )
        
        # Integer encoded representation of the text.
        # 1-D array of shape (text_length,)
        self.text_integer_encoded = self.token_indexer.fit_transform(self.text_tokenized)
        
        # One-hot encoded representation of the text.
        # Aray of shape (text_length n_unique_tokens)
        self.text_one_hot_encoded = self.token_one_hot_encoder.fit_transform(self.text_integer_encoded[:, None])

    def instantiate_model(self):
        """Override to use a different model architecture"""
        batch_size = self.batch_size
        sequence_length = self.sequence_length
        n_tokens = len(self.set_tokens)
        
        return models.get_model_1(
            input_dim=n_tokens,
            output_dim=n_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length
            )

    def _test_data_generator(self):
        """Instantiate a training data generator and
        inspect the shapes of its outputs"""
        # Test the data generator
        data_train = utils.data_dispatcher(
            self.text_integer_encoded, self.text_one_hot_encoded, batch_size=self.batch_size, sequence_length=self.sequence_length
        )
        
        for x, y in data_train:
            print("Shapes of input and target arrays for one batch: ", x.shape, y.shape)
            break
        
    def learn(self, epochs=100, steps_per_epoch=100, validate=False,):
        """Generate training (and optionally validation) data and
        train the model"""
        
        # Instantiate the model
        self.model = self.instantiate_model()
        
        callbacks = []
        validation_data = None
        validation_steps = None
        
        if validate:
            # Last index for training data. The rest
            # will be used for validation
            train_ind = int(0.8 * len(self.text_integer_encoded))
            data_val = utils.data_dispatcher(
                self.text_integer_encoded[train_ind:],
                self.text_one_hot_encoded[train_ind:],
                batch_size=self.batch_size,
                sequence_length=self.sequence_length
            )
            data_train = utils.data_dispatcher(
                self.text_integer_encoded[:train_ind],
                self.text_one_hot_encoded[:train_ind],
                batch_size=self.batch_size,
                sequence_length=self.sequence_length
            )

            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10,
                    monitor="val_accuracy"
                    ),
                mycallbacks.MyModelCheckpoint(
                    export=False,
                    monitor="val_accuracy",
                    mode='max',
                    )
                ]
            validation_data = data_val,
            validation_steps = 20,
            
        else:
            data_train = utils.data_dispatcher(
                self.text_integer_encoded,
                self.text_one_hot_encoded,
                batch_size=self.batch_size,
                sequence_length=self.sequence_length
            )
        
        self.model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.model.fit(
            data_train,
            validation_data=validation_data,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def babble(self, seed, reset=False):
        """
        Synthesize text using the learned model.
        Given a seed string, tokenize and encode the string,
        apply the model to the sequence to initialize its
        state, and then proceed one word at a time, using 
        each single predicted token as the input for the
        next prediction.
        """
        tokenized = np.array(word_tokenize(seed))
        encoded = self.token_indexer.transform(tokenized)
        encoded = np.expand_dims(encoded, 0)
        n_words = encoded.shape[1]
        
        # Whether to reset the states of the
        # stateful LSTM layer before beginning.
        if reset:
            self.model.reset_states()
            
        next_token = ""
        response = seed
        while next_token != "." and len(response) < 600:
        
            X = np.zeros((self.batch_size, n_words), dtype="int")
            X[0,:] = encoded
            pred = self.model(X).numpy()
            
            next_token = "<UNKN>"
            next_token = "<NULL>"
            while next_token == "<NULL>":
                next_token_index = np.argmax(pred[0, -1,:])
                encoded = np.array([[next_token_index]])
                next_token = self.token_indexer.inverse_transform([next_token_index])[0]
                pred[0, -1, next_token_index] = 0
            sep = "" if next_token in [",", ".", "?", "!"] else " "
            response += sep + next_token
        return response

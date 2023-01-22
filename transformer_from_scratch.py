Make Encoder of Transformer
---
Positional Embedding
import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]
sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(
                  output_sequence_length=output_sequence_length,
                  max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)
print("Vocabulary: ", vectorize_layer.get_vocabulary())
print("Vectorized words: ", vectorized_words)
output_length = 6
word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
print(embedded_words)
position_embedding_layer = Embedding(output_sequence_length, output_length)
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
print(embedded_indices)
final_output_embedding = embedded_words + embedded_indices
print("Final output: ", final_output_embedding)
class PositionEmbeddingLayer(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
my_embedding_layer = PositionEmbeddingLayer(output_sequence_length,
                                            vocab_size, output_length)
embedded_layer_output = my_embedding_layer(vectorized_words)
print("Output from my_embedded_layer: ", embedded_layer_output)
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P


    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
Self Attention
import tensorflow as tf
from numpy import random
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer
    def call(self, x):
        x_l1 = self.fully_connected1(x)
        x_l2 = self.fully_connected2(x)
        return self.activation(x_l2)
class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer
    #---#
    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        added = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(added)
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        #h : Number of attention heads.
        # d_k : Size of each attention head for query and key.
        # d_v : Size of each attention head for value.
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
    def call(self, x, padding_mask, training):
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)
        multihead_output = self.dropout1(multihead_output, training=training)
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        return self.add_norm2(addnorm_output, feedforward_output)
# TRANSFORMER ENCODER:
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
    def call(self, input_sentence, padding_mask, training):
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
        x = self.dropout(pos_encoding_output, training=training)
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)
        return x
# # Summary:
# from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
# from multihead_attention import MultiHeadAttention
# from positional_encoding import PositionEmbeddingFixedWeights

# # Implementing the Add & Norm Layer
# class AddNormalization(Layer):
#     def __init__(self, **kwargs):
#         super(AddNormalization, self).__init__(**kwargs)
#         self.layer_norm = LayerNormalization()  # Layer normalization layer

#     def call(self, x, sublayer_x):
#         # The sublayer input and output need to be of the same shape to be summed
#         add = x + sublayer_x

#         # Apply layer normalization to the sum
#         return self.layer_norm(add)

# # Implementing the Feed-Forward Layer
# class FeedForward(Layer):
#     def __init__(self, d_ff, d_model, **kwargs):
#         super(FeedForward, self).__init__(**kwargs)
#         self.fully_connected1 = Dense(d_ff)  # First fully connected layer
#         self.fully_connected2 = Dense(d_model)  # Second fully connected layer
#         self.activation = ReLU()  # ReLU activation layer

#     def call(self, x):
#         # The input is passed into the two fully-connected layers, with a ReLU in between
#         x_fc1 = self.fully_connected1(x)

#         return self.fully_connected2(self.activation(x_fc1))

# # Implementing the Encoder Layer
# class EncoderLayer(Layer):
#     def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
#         super(EncoderLayer, self).__init__(**kwargs)
#         self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
#         self.dropout1 = Dropout(rate)
#         self.add_norm1 = AddNormalization()
#         self.feed_forward = FeedForward(d_ff, d_model)
#         self.dropout2 = Dropout(rate)
#         self.add_norm2 = AddNormalization()

#     def call(self, x, padding_mask, training):
#         # Multi-head attention layer
#         multihead_output = self.multihead_attention(x, x, x, padding_mask)
#         # Expected output shape = (batch_size, sequence_length, d_model)

#         # Add in a dropout layer
#         multihead_output = self.dropout1(multihead_output, training=training)

#         # Followed by an Add & Norm layer
#         addnorm_output = self.add_norm1(x, multihead_output)
#         # Expected output shape = (batch_size, sequence_length, d_model)

#         # Followed by a fully connected layer
#         feedforward_output = self.feed_forward(addnorm_output)
#         # Expected output shape = (batch_size, sequence_length, d_model)

#         # Add in another dropout layer
#         feedforward_output = self.dropout2(feedforward_output, training=training)

#         # Followed by another Add & Norm layer
#         return self.add_norm2(addnorm_output, feedforward_output)

# # Implementing the Encoder
# class Encoder(Layer):
#     def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
#         super(Encoder, self).__init__(**kwargs)
#         self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
#         self.dropout = Dropout(rate)
#         self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

#     def call(self, input_sentence, padding_mask, training):
#         # Generate the positional encoding
#         pos_encoding_output = self.pos_encoding(input_sentence)
#         # Expected output shape = (batch_size, sequence_length, d_model)

#         # Add in a dropout layer
#         x = self.dropout(pos_encoding_output, training=training)

#         # Pass on the positional encoded values to each encoder layer
#         for i, layer in enumerate(self.encoder_layer):
#             x = layer(x, padding_mask, training)

#         return x
Test Encoder of Transformer
---
 
enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack
 
batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
 
input_seq = random.random((batch_size, input_seq_length))
 
encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

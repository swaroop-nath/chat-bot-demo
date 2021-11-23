import numpy as np
from tensorflow import keras
import tensorflow as tf

class PositionalEncoding:
    def trig_encoding(self, pos, idx, val):
        if idx%2 == 0:
            # Sine encoding
            return np.sin(pos/val)
        else:
            #Cosine encoding
            return np.cos(pos/val)
    def __call__(self, pos, d_model):
        encodings_angle = np.array(list(map(lambda idx: np.power(10000, (2 * (idx//2))/d_model), range(d_model))))
        encodings_trig = np.array(list(map(lambda idx_val: self.trig_encoding(pos, *idx_val), enumerate(encodings_angle))))

        return encodings_trig

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_depth = d_model // num_heads

        self.wq = keras.layers.Dense(units=d_model)
        self.wk = keras.layers.Dense(units=d_model)
        self.wv = keras.layers.Dense(units=d_model)

        self.outputs = keras.layers.Dense(units=d_model)

    def _create_mask(self, shape):
        return 1 - np.tri(shape)

    def _scaled_dot_product_attention(self, query, key, value, mask):
        '''
        Input shapes - 
        query: (num_heads, seq_len, head_depth)
        key: (num_heads, seq_len, head_depth)
        value: (num_heads, seq_len, head_depth)
        '''

        dot_prod_attn = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(self.head_depth, tf.float32)
        scaled_dot_prod_attn = dot_prod_attn / tf.math.sqrt(dk)
        # Shape of scaled attn - (num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            # Masking with a really small value, to get 0 prob
            # as attention for these
            miniscule_value_multiplier = -100000
            scaled_dot_prod_attn += mask * miniscule_value_multiplier

        # Softmax applied only on the last axis
        # in order to maintain the fact that attn weights
        # add up to 1 for a single query.
        attn_weights = tf.nn.softmax(scaled_dot_prod_attn, axis=-1)

        outputs = tf.matmul(attn_weights, value)
        # Shape of outputs - (num_heads, seq_len, head_depth)
        return outputs, attn_weights

    def _split_head(self, data):
        '''
        Input shape - (seq_len, d_model)
        Transformed shape - (num_heads, seq_len, head_depth)
        '''
        data = tf.reshape(data, (-1, self.num_heads, self.head_depth))
        return tf.transpose(data, perm=[1, 0, 2])

    def __call__(self, query, key, value, should_mask=False):
        # Do the masking if needed.
        # Get the query, key and value representations and split into heads
        # Do the self attention
        # Concat the results and pass them through the output layer

        '''
        Input shapes - 
        query: (seq_len, d_model)
        key: (seq_len, d_model)
        value: (seq_len, d_model)
        '''
        mask = None
        if should_mask:
            # Mask should be of the shape - (seq_len, seq_len)
            mask = self._create_mask(query.shape[0])

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # Splitting into multiple heads
        query = self._split_head(query)
        key = self._split_head(key)
        value = self._split_head(value)

        attn_output, attn_weights = self._scaled_dot_product_attention(query, key, value, mask) # (num_heads, seq_len, head_depth)
        attn_output = tf.transpose(attn_output, perm=[1, 0, 2]) # (seq_len, num_heads, head_depth)
        concatenated_attn = tf.reshape(attn_output, (-1, self.d_model)) # (seq_len, d_model)

        outputs = self.outputs(concatenated_attn) # (seq_len, d_model)

        return outputs, attn_weights

class FeedForwardNetwork(keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.model = keras.Sequential([
            keras.layers.Dense(units=d_ff, activation='relu'),
            keras.layers.Dense(units=d_model)
        ])
    
    def __call__(self, inputs):
        '''
        input shape == (seq_len, d_model)
        output shape == (seq_len, d_model)
        '''
        return self.model(inputs)

class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        epsilon = 1e-6
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=epsilon)

    def __call__(self, x, prev_key=None, prev_value=None):
        if prev_key == None:
            attn_output, _ = self.attn(x, x, x)
        else:
            # Where recurrence develops
            attn_output, _ = self.attn(x, prev_key, prev_value) # Based on the query, formed from previous encoding, and key and value from current input, attend to current input
        
        norm_output = self.layer_norm_1(x + attn_output)
        ffn_output = self.ffn(norm_output)
        norm_output_2 = self.layer_norm_2(ffn_output + norm_output)

        return norm_output_2

class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        epsilon=1e-6
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm_3 = keras.layers.LayerNormalization(epsilon=epsilon)

    def __call__(self, x, enc_output):
        masked_attn_op, masked_attn_weights = self.masked_attn(x, x, x, should_mask=True)
        norm_output_1 = self.layer_norm_1(masked_attn_op + x)
        enc_dec_attn, cross_attn_weights = self.enc_dec_attn(norm_output_1, enc_output, enc_output) # query comes from the half target output, and key value come from enc_output
        norm_output_2 = self.layer_norm_2(enc_dec_attn + norm_output_1)
        ffn_output = self.ffn(norm_output_2)
        norm_output_3 = self.layer_norm_3(ffn_output + norm_output_2)

        return norm_output_3, masked_attn_weights, cross_attn_weights

class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_ff, num_heads, input_vocab_size):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_dim=input_vocab_size, output_dim=d_model)
        self.pos_encoding = PositionalEncoding()

        self.encoders = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x):
        # Shape of x == (seq_len)
        seq_len = len(x)
        x = self.embedding(x) # --> (seq_len, d_model)
        pos_enc = np.array([self.pos_encoding(i+1, self.d_model) for i in range(seq_len)]) # --> (seq_len, d_model)
        x += pos_enc

        for encoder in self.encoders:
            x = encoder(x)
        return x # --> (seq_len, d_model)

class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_ff, num_heads, output_vocab_size):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(input_dim=output_vocab_size, output_dim=d_model)
        self.pos_encoding = PositionalEncoding()

        self.decoders = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

        self.final_layer = keras.layers.Dense(units=output_vocab_size, activation='softmax') # Final output -> probabilities, not logits


    def predict_once(self, x, enc_output):
        # x -> The output prepended and appended with [START] and [END] tokens
        # shape of x --> (seq_len, )
        # enc_output -> the output from the encoder
        # shape of enc_output --> (seq_len, d_model)
        seq_len = len(x)
        x = self.embedding(x) # --> (seq_len, d_model)
        pos_enc = np.array([self.pos_encoding(i+1, self.d_model) for i in range(seq_len)]) # --> (seq_len, d_model)
        x += pos_enc

        masked_weights = []
        cross_weights = []
        for decoder in self.decoders:
            x, masked_attn_weights, cross_attn_weights = decoder(x, enc_output)
            masked_weights.append(masked_attn_weights)
            cross_weights.append(cross_attn_weights)

        outputs = self.final_layer(x) # --> (seq_len, output_vocab_size) -> outputs prob for each place
        return outputs, masked_weights[0], cross_weights[-1]

class Transformer(keras.Model):
    def __init__(self, num_enc_layers, num_dec_layers, d_model, d_ff, num_heads, input_vocab_size, output_vocab_size):
        super().__init__()
        self.encoder = Encoder(num_enc_layers, d_model, d_ff, num_heads, input_vocab_size)
        self.decoder = Decoder(num_dec_layers, d_model, d_ff, num_heads, output_vocab_size)

    def __call__(self, inputs):
        # inp_sent is basically a single prompt
        # tar_half_sent is the sent that has been generated up until now
        inp_sent, tar_half_sent = inputs

        enc_output = self.encoder(inp_sent)

        dec_output, masked_attn_weights, cross_attn_weights = self.decoder.predict_once(tar_half_sent, enc_output)
        # returns probability for all the tokens in the window slid forward by 1 unit
        # thus, the tar real has to be the real tokens in the window slid by 1 unit
        # the probability on all the tokens in this window will be used for cross entropy computation
        return dec_output, masked_attn_weights, cross_attn_weights

    def predict(self, inp_sent, dec_prompt, stop_id, temperature, max_len=15):
        # inp_sent = list, dec_prompt = list
        inp_sent = np.array(inp_sent)
        dec_prompt = np.asarray(dec_prompt)

        prev_output = -1
        pred_seq = []
        iter = 0

        while prev_output != stop_id and iter <= max_len:
            inputs = (inp_sent, dec_prompt)
            dec_output, masked_attn_weights, cross_attn_weights = self.__call__(inputs)
            dec_output = dec_output.numpy() # Can't do this!
            probs = self.apply_temp(dec_output[-1, :], temperature)
            pred = np.argmax(np.random.multinomial(1, probs, 1))
            prev_output = pred
            dec_prompt = np.append(dec_prompt, prev_output).astype(np.int32)
            pred_seq.append(pred)
            iter += 1

        return pred_seq, masked_attn_weights, np.around(tf.math.reduce_mean(cross_attn_weights, axis=0).numpy(), decimals=3)

    def apply_temp(self, probs, temp):
        # return probs
        mod_logits = tf.math.log(probs)/temp
        mod_probs = tf.math.exp(mod_logits)
        return mod_probs / tf.math.reduce_sum(mod_probs)
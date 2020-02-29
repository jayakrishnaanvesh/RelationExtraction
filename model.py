import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.layer = layers.Bidirectional(tf.keras.layers.GRU(units=hidden_size, activation="tanh",return_state=False, use_bias=True,return_sequences=True))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        M= tf.tanh(rnn_outputs)
        alpha=tf.nn.softmax(tf.matmul(M,self.omegas),axis=1)
        r=tf.reduce_sum(tf.multiply(alpha,rnn_outputs),axis=[1])
        ### TODO(Students) END
        output=tf.tanh(r)
        return output

    def call(self, inputs, pos_inputs, training):
        tokens_mask = tf.cast(inputs != 0, tf.float32)
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        input=tf.concat([word_embed,pos_embed],axis=-1)
        current_sequence=self.layer(inputs=word_embed,training=training,mask=tokens_mask)
        attn_output=self.attn(current_sequence)
        logits=self.decoder(attn_output)
        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self,batch_size: int , vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()

        self.num_classes = len(ID_TO_CLASS)
        self.omegas = tf.Variable(tf.random.normal((hidden_size, 1)))
        self.decoder1 = layers.Dense(units=self.num_classes)
        self.decoder2 = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.fw_layer = tf.keras.layers.GRU(units=hidden_size, activation="tanh",  use_bias=True,
                                return_sequences=True)
        self.bw_layer = tf.keras.layers.GRU(units=hidden_size, activation="tanh",  use_bias=True,
                                return_sequences=True)
        self.combined_layer=tf.keras.layers.GRU(units=hidden_size, activation="tanh", use_bias=True,
                                return_sequences=True,return_state=True)
        self.conv_layer = layers.Conv1D(200, 1, activation='tanh', input_shape=(batch_size, None, None))
        self.maxpool1 = layers.MaxPooling1D(3)
        self.dropout1 = layers.Dropout(0.4)
        self.flatten = layers.Flatten()
        self.globalPool = layers.GlobalAveragePooling1D()

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        M= tf.tanh(rnn_outputs)
        alpha=tf.nn.softmax(tf.matmul(M,self.omegas),axis=1)
        r=tf.reduce_sum(tf.multiply(alpha,rnn_outputs),axis=[1])
        ### TODO(Students) END
        output=tf.tanh(r)
        return output
    def call(self, inputs, pos_inputs, training):
        tokens_mask = tf.cast(inputs != 0, tf.float32)
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        f_seq=self.fw_layer(inputs=word_embed,training=training,mask=tokens_mask)
        rev_input=tf.reverse(word_embed,[1])
        b_seq=self.bw_layer(inputs=rev_input,training=training,mask=tokens_mask)
        seq_conc=tf.concat([b_seq,f_seq],2)
        combined_seq,state=self.combined_layer(inputs=seq_conc,training=training,mask=tokens_mask)
        # final representation is obtained by maxpooling the outputs of recurrent neural networks
        #final_rep=tf.reduce_max(combined_seq,axis=1)
        final_rep=self.attn(combined_seq)
        logits1=self.decoder1(final_rep)
        output = self.dropout1(self.maxpool1(self.conv_layer(word_embed, training=training)))
        output = self.globalPool(output)
        logits2 = self.decoder2(output)
        #a=tf.Variable(0.9,trainable=True)
        logits=tf.add(tf.scalar_mul(0.9,logits1),tf.scalar_mul(0.1,logits2))
        return {'logits': logits}

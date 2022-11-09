
import tensorflow as tf
from tensorflow.keras.layers import  Conv1D, Lambda,Reshape, Flatten, UpSampling1D, Dense, AveragePooling1D
from datetime import datetime
from keras import backend as K



def createVAEModel(ih,iw,latent_dim):

    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon



    # =============================
    # Encoder Model
    inputs = tf.keras.layers.Input((ih,iw), name="rgb")

    x = Conv1D(12,3, activation='tanh', padding='same',dilation_rate=2)(inputs)
    x1 = AveragePooling1D(2)(x)
    x2 = Conv1D(8,3, activation='tanh', padding='same',dilation_rate=2)(x1)
    x3 = AveragePooling1D(2)(x2)
    x2 = Conv1D(4,3, activation='tanh', padding='same',dilation_rate=2)(x2)
    x4 = AveragePooling1D(2)(x2)
    h = Flatten()(x4)

    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    z = Lambda(sampling)([z_mean, z_log_sigma])

    # =============================
    # Decoder Model
    l = tf.keras.layers.Input(latent_dim, name="latent")
    d1 = Dense(128)(l)
    d2 = Reshape((32,4))(d1)

    d3 = Conv1D(4,1,strides=1, activation='tanh', padding='same')(d2)
    d4 = UpSampling1D(2)(d3)

    d40 = Conv1D(4,1,strides=1, activation='tanh', padding='same')(d4)
    d5 = UpSampling1D(2)(d40)

    d5 = Conv1D(8,1,strides=1, activation='tanh', padding='same')(d5)
    d6 = UpSampling1D(2)(d5)

    decoded = Conv1D(4,1,strides=1, activation='linear', padding='same')(d6)



    encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_sigma, z], name='encoder')
    decoder = tf.keras.Model(inputs=[l], outputs=[decoded], name='decoder')


    # =============================
    # VAE Model
    outputs = decoder(encoder(inputs)[2])

    model = tf.keras.Model(inputs, outputs, name='autoencoder')

    # =============================
    # Add VAE loss

    reconstruction_loss = 0.5*K.sum(K.square(inputs-outputs))/0.01

    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)

    return model, encoder, decoder

import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv1D, Lambda, MaxPooling1D, Reshape, BatchNormalization, Flatten, UpSampling1D, Dense, AveragePooling1D
from datetime import datetime
from tensorflow.keras import regularizers
from keras import backend as K

from training.datapipe import Datapipe
from training.callbacks import DrawImageCallback
from training.model import createVAEModel
from config import parse_args



def main(arg):


    ih, iw = arg.ih, 4
    learnrate = arg.learnRate
    batchSize = arg.batchSize



    dp = Datapipe("../data/processed/*.json")
    g, gt = dp.create(split= arg.testTrainSplit, batchSize=batchSize)

    latent_dim = arg.latentDim


    # =============================
    # VAE Model

    model, encoder, decoder = createVAEModel(ih,iw,latent_dim)

    print(model.summary())

    opti = tf.keras.optimizers.Adam(learnrate)

    model.compile(optimizer=opti)


   #model.load_weights("weights_cpk.h5")


    # ================================
    now = datetime.now()
    #timestamp = datetime.timestamp(now)
    timestamp = now.strftime('%Y-%m-%d-%H%M%S')

    drwcb = DrawImageCallback(logdir=f"./tblogs/ae/{timestamp}",tfdataset=gt, encoder=encoder)


    tfbcb = tf.keras.callbacks.TensorBoard(
        log_dir=f"./tblogs/ae/{timestamp}", histogram_freq=0, write_graph=True,
        write_images=True, update_freq='batch',
        profile_batch=2, embeddings_freq=0, embeddings_metadata=None
    )

    estcb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=140, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    mcpcb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join('weights_cpk.h5'), monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch',
    )

    rlrcb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=35,
        verbose=0,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    term = tf.keras.callbacks.TerminateOnNaN()

    # ================================


    model.fit(
        g, epochs=arg.epochs,
        callbacks = [tfbcb, mcpcb, estcb, rlrcb, term, drwcb],
        validation_data=gt
    )

    model.save_weights("weights_final.h5")

if __name__ == "__main__":
    arg = parse_args()
    main(arg)
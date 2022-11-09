import os
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np






class DrawImageCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        logdir,
        tfdataset,
        encoder,
        writerName="imager",
    ):
        super(DrawImageCallback, self).__init__()

        self.tbcb = tf.summary.create_file_writer(os.path.join(logdir, writerName))
        self.writerName = writerName
        self.step_number = 0
        self.tfdataset = tfdataset
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        """Draw images at the end of an epoche

        Args:
            epoch ([type]): [description]
            logs ([type], optional): [description]. Defaults to None.
        """


        latent = []
        for (x, ytrue) in self.tfdataset:
            latent.append(self.encoder.predict(x)[2])

        latent = np.vstack(latent)


        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)

        # Do some plotting here
        ax = fig.add_subplot(111)

        ax.scatter(latent[:,0], latent[:,1]) # latent[:,2]

        ax.grid(True)
   
        # Retrieve a view on the renderer buffer
        canvas.draw()
        latentImgs = np.expand_dims(np.asarray(canvas.buffer_rgba()),0)
        # convert to a NumPy array




        x, ytrue = None, None


        for (x, ytrue) in self.tfdataset:
            ypred = self.model.predict(x)

            break


        imgs = []

        for i in range(x.shape[0]):
            # make a Figure and attach it to a canvas.
            fig = Figure(figsize=(5, 4), dpi=100)
            canvas = FigureCanvasAgg(fig)

            # Do some plotting here
            ax = fig.add_subplot(111)


            ax.plot(ytrue[i,:,0], ytrue[i,:,1],'b--')
            ax.plot(ytrue[i,:,2], ytrue[i,:,3],'r--')

            ax.plot(ypred[i,:,0], ypred[i,:,1],'b-')
            ax.plot(ypred[i,:,2], ypred[i,:,3],'r-')
            ax.grid(True)
            ax.axis([-0.5*0.1,0.5*0.1,-0.3*0.1,0.7*0.1])
            # Retrieve a view on the renderer buffer
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            # convert to a NumPy array
            imgs.append(np.expand_dims(buf,0))

        imgs = np.vstack(imgs)


        with self.tbcb.as_default():

            tf.summary.image(
                "Images", imgs[...,:3], max_outputs=25, step=self.step_number
            )
            tf.summary.image(
                "LatentSpace", latentImgs[...,:3], max_outputs=25, step=self.step_number
            )


        self.step_number += 1
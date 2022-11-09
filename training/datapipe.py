import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import numpy as np






class Datapipe:
    def __init__(self, path):


        self.filenames = glob.glob(path)


    # =================================================
    @staticmethod
    def readJson(jsonfile):


        with open(jsonfile, 'r') as f1:
            data = json.load(f1)


        name = data["name"]
        ss = np.asarray(data["ss"], dtype=np.float32)
        ps = np.asarray(data["ps"], dtype=np.float32)


        geom = np.vstack((ss,ps)).T

        return name, geom

    # =================================================
    def autoencoder(self, name, geom):
        return geom, geom

    def scaleer(self, name, geom):

        geom = geom*tf.constant([1.0, 5.0, 1.0, -5.0], dtype=tf.float32) + tf.constant([[-0.5, 0.0, -0.5, -0.0]], dtype=tf.float32) 
        geom = geom*0.1
        
        return name, geom
    # =================================================
    def create(self, split=0.1, batchSize=10):

        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)

        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.repeat(1)

        # Load the Json
        dataset = dataset.map(self._loadJson)

        dataset = dataset.map(self.scaleer)
        dataset = dataset.map(self.autoencoder)


        train_size = int((1-split) * len(self.filenames))


        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)


        train_dataset = train_dataset.batch(batchSize)
        test_dataset = test_dataset.batch(batchSize)

        return train_dataset, test_dataset

    # ============================
    def _loadJson(self, jsonfile):

        def _pyLoadJson(content):
            return Datapipe.readJson(
                jsonfile=content.numpy().decode("utf-8"),
            )
      

        name, geom = tf.py_function(
            _pyLoadJson, [jsonfile], [tf.string, tf.float32]
        )

        return name, geom



if __name__ == "__main__":


    dp = Datapipe("database_processed/*.json")

    g, gt = dp.create(split=0.3)


    for (x,y) in g.take(4):
        print(x)
        print(y)


        plt.plot(y[0,:,0], y[0,:,1],'b-')
        plt.plot(y[0,:,2], y[0,:,3],'r-')

        plt.show()

        plt.imshow(y[0,...], aspect="auto")
        
        plt.show()
'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
from time import time
from yelp_multiclass.data.yelp_dataset import load_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import TensorBoard

from yelp_multiclass.data.config import getModelFile,getDataFile



class TensorBoard_Batch_Stats(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)


max_features = 100000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32
npz_file=getDataFile()
print('Loading data from :: '+npz_file)
(x_train, y_train), (x_test, y_test) = load_data(path=npz_file,num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print(y_train[20])

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard_Batch_Stats(log_dir="G:/tensorboard_logs/{}".format(time()),histogram_freq=1, batch_size=batch_size)

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          verbose=1,
          validation_split=0.2,
          callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_file=getModelFile()
model.save(model_file)
print("Saved model to :: "+model_file)
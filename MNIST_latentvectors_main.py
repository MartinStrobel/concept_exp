import os
from tensorflow import keras
from tensorflow.keras  import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D, Dropout, Layer
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from absl import app
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import gc 
import argparse

import tensorflow_datasets as tfds
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-seedL', nargs=1, type=int, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-seedH', nargs=1, type=int, default='30') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)
init = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

def create_org_model(dataset, width=28, 
                       height=28, channel=1, verbose=True,epochs=10, load=False):
    """Train a base model"""
    input1 = Input(
      shape=(
          width,
          height,
          channel,
      ), name='concat_input')
    conv1 = Conv2D(32, kernel_size=5, activation='relu', padding='same')
    conv2 = Conv2D(32, kernel_size=5, activation='relu', padding='same')
    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')
    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')
    dense1 = Dense(256, activation='relu')
    predict = Dense(10, activation='softmax')

    conv1o = conv1(input1)
    conv2o = conv2(conv1o)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2o)
    drop1  = Dropout(.25)(pool1)
    conv3o = conv3(drop1)
    conv4o = conv4(conv3o)
    pool2 = MaxPooling2D(pool_size=(2, 2),  strides=(2,2))(conv4o)
    drop2  = Dropout(.25)(pool2)
    drop2f = Flatten()(drop2)
    fc1 = dense1(drop2f)
    softmax1 = predict(fc1)

    drop2_2 = Input(shape=(7,7,64), name='concat_input')  
    drop2f_2 = Flatten()(drop2_2)
    fc1_2 = dense1(drop2f_2)
    softmax1_2 = predict(fc1_2)

    mlp = Model(input1, softmax1)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    mlp.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'])

    if load:
      mlp.load_weights(model_dir+'complete_model.h5')
    else:
      mlp.fit(dataset, epochs=epochs,verbose=verbose)
      mlp.save_weights(model_dir+'complete_model.h5')

    for layer in mlp.layers:
        layer.trainable = False

    feature_model = Model(input1, drop2)
    predict_model  = Model(drop2_2, softmax1_2)

    return feature_model, predict_model, mlp

def new_training_set(seed,percentage):
    np.random.seed(seed)
    indices = np.random.choice(len(x_train), size=int(percentage*len(x_train)), replace=False)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train[indices], y_train[indices]))
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(len(x_train))
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, indices

class Weight(Layer):
    """Simple Weight class."""

    def __init__(self, dim, **kwargs):
        self.dim = dim
        super(Weight, self).__init__(**kwargs)

    def build(self, input_shape):
        # creates a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='proj', shape=self.dim, initializer=init, trainable=True)
        super(Weight, self).build(input_shape)

    def call(self, x):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return self.dim

def mean_sim(topic_prob_n,n_concept):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return 1*tf.reduce_mean(input_tensor=tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=32,sorted=True).values)
  return loss

def topic_model_new_MNIST(predict,
           f_train,
           y_train,
           n_concept,
           verbose=False,
           metric1=['accuracy'],
           opt='adam',
           loss1=tf.nn.softmax_cross_entropy_with_logits,
           thres=0.0,
           load=False):
    """Returns main function of topic model."""


    f_input = Input(shape=(f_train.shape[1],f_train.shape[2],f_train.shape[3]), name='f_input')
    f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(f_input)

    topic_vector = Weight((f_train.shape[3], n_concept))(f_input)
    topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)
    topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
    topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
    topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
    topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
    topic_prob_sum = Lambda(lambda x: K.sum(x, axis=3, keepdims=True)+1e-3)(topic_prob_am)
    topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])

    rec_vector_1 = Weight((n_concept, 500))(f_input)
    rec_vector_2 = Weight((500, f_train.shape[3]))(f_input)
    rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
    rec_layer_2 = Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
    pred = predict(rec_layer_2)
    topic_model_pr = Model(inputs=f_input, outputs=pred)
    topic_model_pr.layers[-1].trainable = False
    if opt =='sgd':
        optimizer = SGD(lr=0.001)
        optimizer_state = [optimizer.iterations, optimizer.lr,
              optimizer.momentum, optimizer.decay]
        optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
    elif opt =='adam':
        optimizer = Adam(lr=0.001)
        optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
                                 optimizer.beta_2, optimizer.decay]
        optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
    metric1.append(mean_sim(topic_prob_n, n_concept))
    topic_model_pr.compile(
      loss=topic_loss_MNIST(topic_prob_n, topic_vector_n,  n_concept, f_input, loss1=loss1),
      optimizer=optimizer,metrics=metric1)
    print(topic_model_pr.summary())
    if load:
        topic_model_pr.load_weights(load)
    return topic_model_pr, optimizer_reset, optimizer, topic_vector_n,  n_concept, f_input


def topic_loss_MNIST(topic_prob_n, topic_vector_n, n_concept, f_input, loss1):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return (1.0*tf.reduce_mean(input_tensor=loss1(y_true, y_pred))\
            - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=32,sorted=True).values))
            + 0.1*tf.reduce_mean(input_tensor=(K.dot(K.transpose(topic_vector_n), topic_vector_n) - tf.eye(n_concept)))
            )
  return loss

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


base_architecture = 'new_MNIST_base'
percentage = 0.7
verbose = 2
n_concept = 10
batch_size = 128
org_epochs = 10
topic_epochs = 10

train = pd.read_csv("../data/MNIST/mnist_train.csv")
test = pd.read_csv("../data/MNIST/mnist_test.csv")

y_train = train["label"]
y_test = test["label"]
# Drop 'label' column
x_train = train.drop(labels = ["label"],axis = 1) 
x_train = x_train.values.reshape(-1,28,28,1)
x_test = test.drop(labels = ["label"],axis = 1) 
x_test = x_test.values.reshape(-1,28,28,1)

for experiment_run in range(args.seedL[0],args.seedH[0]):
  seed = int(experiment_run)
  experiment_run = str(experiment_run)
  model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
  makedir(model_dir)
  ds_train_small, ind = new_training_set(seed,percentage)
  feature_model, predict_model, model = create_org_model(ds_train_small,  verbose=verbose, epochs=org_epochs, load=True)

  train_predict = model.predict(x_train.astype('float32')).argmax(axis=1)
  test_predict = model.predict(x_test.astype('float32')).argmax(axis=1)


  f_train = feature_model.predict(x_train)
  f_test = feature_model.predict(x_test)

  pickle.dump(f_train, open(model_dir+"full_train_latent_{}.pkl".format(seed),"wb"))
  pickle.dump(f_test, open(model_dir+"full_test_latent_{}.pkl".format(seed),"wb"))

  topic_model_pr, optimizer_reset, optimizer, topic_vector,  n_concept, f_input = topic_model_new_MNIST(predict_model,
                                                                      f_train,
                                                                      y_train[ind],
                                                                      n_concept,
                                                                      verbose=verbose,
                                                                      metric1=['accuracy'],
                                                                      loss1=tf.keras.losses.sparse_categorical_crossentropy,
                                                                      thres=0.2,
                                                                      load=False)



  topic_model_pr.load_weights(model_dir+'latest_topic_MNIST.h5')

  topic_vec = topic_model_pr.layers[1].get_weights()[0]
  pickle.dump(topic_vec, open(model_dir+"topic_latent_{}.pkl".format(seed),"wb"))

import os
from tensorflow import keras
from tensorflow.keras  import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D, AveragePooling2D, Dropout, Layer

from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from absl import app
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import gc 
import argparse
from ResNet34 import ResNet34

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

def create_org_model(train_generator, val_generator, input_shape=(224, 224, 3),  verbose=True):
    """Loads pretrain model or train one."""
    convlayers = ResNet34(input_shape=input_shape, input_tensor=None, weights='imagenet', include_top=False)
    dense1 = Dense(200,activation='softmax')

    model=Sequential()
    model.add(convlayers)
    model.add(AveragePooling2D(7,7))
    model.add(Flatten())
    model.add(dense1)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    opt=tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=opt)
    
    history=model.fit_generator(train_generator,validation_data=val_generator,
         epochs=10, verbose=verbose)
    
    for layer in model.layers:
        layer.trainable = False
    
    feature_model =  Sequential()
    feature_model.add(convlayers)
    feature_model.add(AveragePooling2D(7,7))
    predict_model  =  Sequential()
    predict_model.add(Flatten())
    predict_model.add(dense1)

    return feature_model, predict_model, model

def new_train_generator(seed,percentage,df,org_datasetsize,repetition=30):
    np.random.seed(seed)
    indices = np.random.choice(org_datasetsize, size=int(percentage*org_datasetsize), replace=False)
    augmented_indices = indices.repeat(repetition)*repetition+np.tile(np.arange(repetition),len(indices))
    train_generator =train_datagen.flow_from_dataframe(dataframe=df,
        directory=train_directory,
        x_col="x_col",
        y_col="y_col",
        target_size=(224,224),
        color_mode='rgb',
        shuffle=True,
        seed=seed,
        class_mode='sparse',batch_size=batch_size,
    )
    return train_generator, indices

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

def topic_model_new_BIRD(predict,
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


def topic_loss_BIRD(topic_prob_n, topic_vector_n, n_concept, f_input, loss1):
  """creates loss for topic model"""
  def loss(y_true, y_pred):
    return (1.0*tf.reduce_mean(input_tensor=loss1(y_true, y_pred))\
            - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=32,sorted=True).values))
            + 0.1*tf.reduce_mean(input_tensor=(K.dot(K.transpose(topic_vector_n), topic_vector_n) - tf.eye(n_concept)))
            )
  return loss


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


base_architecture = 'BIRD_base'
percentage = 0.7
verbose = 2
n_concept = 10
batch_size = 128
org_epochs = 1
topic_epochs = 10

train_directory='../data/cup_200_cropped/train_cropped_augmented2'
org_train_directory='../data/cup_200_cropped/train_cropped'
test_directory='../data/cup_200_cropped/test_cropped'

train_datagen=ImageDataGenerator(rescale=1/255)

# Create the dataframe to sample from the augmented dataset
filenames = []
folders = os.listdir(train_directory)
for foldername in sorted(folders):
    files = os.listdir(train_directory+"/"+foldername)
    for filename in sorted(files):
        filenames.append(foldername+"/"+filename)
df = pd.DataFrame(filenames ,columns=["x_col"])
df["y_col"] = df["x_col"].map(lambda x: x.split(".")[0])


# Create the dataframe for the original dataset
filenames = []
folders = os.listdir(org_train_directory)
for foldername in sorted(folders):
    files = os.listdir(org_train_directory+"/"+foldername)
    for filename in sorted(files):
        filenames.append(foldername+"/"+filename)
org_df = pd.DataFrame(filenames ,columns=["x_col"])
org_df["y_col"] = org_df["x_col"].map(lambda x: x.split(".")[0])

org_train_generator =train_datagen.flow_from_dataframe(dataframe=org_df,
    directory=train_directory,
    x_col="x_col",
    y_col="y_col",
    target_size=(224,224),
    color_mode='rgb',
    shuffle=False,
    class_mode='sparse',batch_size=batch_size,
)

# Create the dataframe for the test dataset
filenames = []
folders = os.listdir(test_directory)
for foldername in sorted(folders):
    files = os.listdir(test_directory+"/"+foldername)
    for filename in sorted(files):
        filenames.append(foldername+"/"+filename)
test_df = pd.DataFrame(filenames ,columns=["x_col"])
test_df["y_col"] = test_df["x_col"].map(lambda x: x.split(".")[0])

test_generator =train_datagen.flow_from_dataframe(dataframe=test_df,
    directory=train_directory,
    x_col="x_col",
    y_col="y_col",
    target_size=(224,224),
    color_mode='rgb',
    shuffle=False,
    class_mode='sparse',batch_size=batch_size,
)

for experiment_run in range(args.seedL[0],args.seedH[0]):
  seed = int(experiment_run)
  experiment_run = str(experiment_run)
  model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
  makedir(model_dir)
  train_generator, indices = new_train_generator(seed,percentage,df,org_datasetsize=len(org_df),repetition=30)
  feature_model, predict_model, model = create_org_model(train_generator,val_generator)

  train_predict = model.predict(org_train_generator).argmax(axis=1)
  test_predict = model.predict(test_generator).argmax(axis=1)
  pickle.dump(train_predict, open(model_dir+"full_train_{}.pkl".format(seed),"wb"))
  pickle.dump(test_predict, open(model_dir+"full_test_{}.pkl".format(seed),"wb"))

  batches = 0 
  f_train = []
  f_train_y = []
  for x_batch, y_batch in train_generator:
    f_train_batch = feature_model.predict(x_batch)
    f_train.append(f_train_batch)
    f_train_y.append(y_batch)
    batches += 1
    print(batches, end=', ')
    if batches >= len(train_generator):
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break
  f_train = np.concatenate(f_train)
  f_train_y = np.concatenate(f_train_y)

  topic_model_pr, optimizer_reset, optimizer, topic_vector,  n_concept, f_input = topic_model_new_BIRD(predict_model,
  																		                                f_train,
  																		                                f_train_y,
  																		                                n_concept,
  																		                                verbose=verbose,
  																		                                metric1=['accuracy'],
  																		                                loss1=tf.keras.losses.sparse_categorical_crossentropy,
  																		                                thres=0.2,
  																		                                load=False)
  topic_model_pr.fit(f_train,f_train_y,batch_size=batch_size,epochs=topic_epochs, verbose=verbose)

  f_train_full = feature_model.predict(org_train_generator)
  f_test_full = feature_model.predict(test_generator)
  train_predict_con = topic_model_pr.predict(f_train_full).argmax(axis=1)
  test_predict_con = topic_model_pr.predict(f_test_full).argmax(axis=1)

  pickle.dump(train_predict_con, open(model_dir+"con_train_{}.pkl".format(seed),"wb"))
  pickle.dump(test_predict_con, open(model_dir+"con_test_{}.pkl".format(seed),"wb"))

  topic_model_pr.save_weights(model_dir+'latest_topic_BIRD.h5')

  topic_vec = topic_model_pr.layers[1].get_weights()[0]
  recov_vec = topic_model_pr.layers[-3].get_weights()[0]
  topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)

  f_train_n = f_train/(np.linalg.norm(f_train,axis=3,keepdims=True)+1e-9)
  topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
  topic_prob = np.matmul(f_train_n,topic_vec_n)
  n_size = 7
  j_int, a, b = {},{},{}
  for i in range(n_concept):
      j_int[i], a[i], b[i] = [], [], []
      ind = np.argpartition(topic_prob[:,:,:,i].flatten(), -10)[-10:]
      sim_list = topic_prob[:,:,:,i].flatten()[ind]
      for jc,j in enumerate(ind):
          j_int[i].append(int(np.floor(j/(n_size*n_size))))
          a[i].append(int((j-j_int[i][-1]*(n_size*n_size))/n_size))
          b[i].append(int((j-j_int[i][-1]*(n_size*n_size))%n_size))
  pickle.dump(a, open(model_dir+"a_{}.pkl".format(seed),"wb"))
  pickle.dump(b, open(model_dir+"b_{}.pkl".format(seed),"wb"))
  pickle.dump(j_int, open(model_dir+"j_int_{}.pkl".format(seed),"wb"))
  collected = gc.collect()
  print("Garbage collector: collected", "%d objects." % collected) 


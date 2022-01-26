

import numpy as np
import tensorflow as tf
from. import dataset
from . import losses
from . import metrics
from .models.all_models import model_from_name
'''
eg.
model_from_name["unet"] = unet
model_from_name["highresnet"] = highresnet
'''



class base_class():
  

class Segmentation(base_class):
  """
  Segmentation API. 
  
  Sequentially applies transforms, train, predict 
  and evaluate the segmentation model. 
  
  """
  def __init__(
    self, 
    n_classes,
    input_shape,
    activation="relu",
    batchnorm=False,
    batch_size=None,
    model_name='unet',
    multi_gpu=False, 
    learning_rate=1e-04,
  ):
    self.n_classes = n_classes, 
    self.input_shape = input_shape 
    self.activation = activation
    self.batchnorm= batchnorm
    self.model_name= model_name
    self.multi_gpu= multi_gpu
    self.learning_rate=learning_rate
    
  def _fit(self):
    model = model_from_name[self.model_name](self.n_classes, 
                                      (*self.input_shape,1),
                                      activation = self.activation
                                      batchnorm= self.batchnorm)
    model.compile(tf.keras.optimizers.Adam(self.learning_rate),
      loss=losses.dice,
      metrics=[metrics.dice, metrics.jaccard])
    return model
    
  def fit(self,
          data, 
          train_epochs=1,
          train_epoch_steps=1,
          val_epoch_steps=1, 
          **kwds): 
    #estimate or train a model
   if self.multi_gpu:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      model = _fit(self)
   else:
    model = _fit(self)
   #model.fit would confuse with the segmentation.fit ? 
   model.fit(
    data[0],
    epochs,
    steps_per_epoch=steps_per_epoch, 
    validation_data=data[1], 
    validation_steps=validation_steps)
    
  def fit_transform():
    #train and apply estimates to input
  def transform():
    #transform inputs using estimates
  def load():
    #load a saved model
  def predict():
    #apply trained model to input
  def save():
  def evaluate():
    # evaluate using a trained model, test inputs and labels.

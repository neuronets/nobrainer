

import numpy as np
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
  
  def fit(self, x, y, **kwds): 
    #estimate or train a model
    
   model = model_from_name[model_name](self.n_classes, 
                                      (*self.input_shape,1),
                                      activation = self.activation
                                      batchnorm= self.batchnorm)
   model.compile(
    optimizer=optimizer,
    loss=nobrainer.losses.dice,
    metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard])
  
   
    
    
    
    
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

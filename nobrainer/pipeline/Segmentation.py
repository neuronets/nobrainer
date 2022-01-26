

import numpy as np

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
  ):
    self.n_classes = n_classes, 
    self.input_shape = input_shape 
    self.activation = activation
    self.batchnorm= batchnorm
    self.model_name= model_name
  
  def fit(self, x, y, **kwds): 
    #estimate or train a model
    
    
    
    
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

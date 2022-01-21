import tensorflow as tf
from tensorflow.keras import Model

Layers = tf.keras.layers

def nnUNet(n_classes, 
           input_shape, 
           feature_maps=32,
           max_feature_maps=480, 
           num_pool=8, 
           kernel_initializer='he_normal', 
           batch_size = None):
    
    """ 3D Adaptation of `nnU-Net Github <https://github.com/MIC-DKFZ/nnUNet`.
                                                                                
       Parameters
       ----------
       n_classes(int): number of classes.
       image_shape(tuple): 3D tuple Dimensions of the input image.              
       feature_maps(int): feature maps to start with in the 
       first level of the U-Net. 
       max_feature_maps(int): number of maximum feature maps
       allowed to used in conv layers.
       num_pool(int): number of pooling (downsampling) operations.
       kernel_initializer(str):kernel initialization for convolutional layers.                                                         
                                                                          
       Returns
       -------                                                                 
       Model object.              
    """
                       
    x = Layers.Input(shape=input_shape, batch_size=batch_size) 
    inputs = x                                             
    ls=[]
    seg_outputs = []
    feature_save = []

    # ENCODER
    x = StackedConvLayers(x, feature_maps, kernel_initializer, first_conv_stride=1)
    feature_save.append(feature_maps)
    feature_maps = feature_maps*2 if feature_maps*2 < max_feature_maps else max_feature_maps
    ls.append(x)

    # conv_blocks_context
    for i in range(num_pool-1):
        x = StackedConvLayers(x, feature_maps, kernel_initializer)
        feature_save.append(feature_maps)
        feature_maps = feature_maps*2 if feature_maps*2 < max_feature_maps else max_feature_maps
        ls.append(x)

    # BOTTLENECK
    x = StackedConvLayers(x, feature_maps, kernel_initializer, first_conv_stride=(1,2))

    # DECODER
    for i in range(len(feature_save)):
        # tu
        if i == 0:
            x = Layers.Conv3DTranspose(feature_save[-(i+1)], (1, 2), use_bias=False,
                                strides=(1, 2), padding='valid') (x)
        else:
            x = Layers.Conv3DTranspose(feature_save[-(i+1)], (2, 2), use_bias=False,
                                strides=(2, 2), padding='valid') (x)
        x = Layers.concatenate([x, ls[-(i+1)]])

        # conv_blocks_localization
        x = StackedConvLayers(x, feature_save[-(i+1)], kernel_initializer, first_conv_stride=1)
        seg_outputs.append(Layers.Conv3D(n_classes, (1, 1), use_bias=False, activation="softmax") (x))   

    outputs = seg_outputs
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def StackedConvLayers(x, feature_maps, kernel_initializer, first_conv_stride=2):
    x = ConvDropoutNormNonlin(x, feature_maps, kernel_initializer, 
                              first_conv_stride=first_conv_stride)
    x = ConvDropoutNormNonlin(x, feature_maps, kernel_initializer)
    return x

    
def ConvDropoutNormNonlin(x, feature_maps, k_init, first_conv_stride=1):
    x = Layers.ZeroPadding3D(padding=(1, 1, 1))(x)
    x = Layers.Conv3D(feature_maps, (3, 3), strides=first_conv_stride,
                      activation=None,
                      kernel_initializer=k_init, 
                      padding='valid') (x)
    x = Layers.BatchNormalization(epsilon=1e-05, momentum=0.1) (x)
    x = Layers.LeakyReLU(alpha=0.01) (x)
    return x

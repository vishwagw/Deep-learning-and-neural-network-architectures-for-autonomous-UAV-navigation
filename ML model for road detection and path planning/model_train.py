import tensorflow as tf
from tf_keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tf_keras.models import Model

def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder (EfficientNet-B3)
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        input_tensor=inputs
    )
    
    # Decoder
    c1 = backbone.get_layer('block2a_expand_activation').output
    c2 = backbone.get_layer('block3a_expand_activation').output
    c3 = backbone.get_layer('block4a_expand_activation').output
    c4 = backbone.get_layer('block6a_expand_activation').output
    
    # Upsampling path
    u1 = UpSampling2D()(c4)
    u1 = Concatenate()([u1, c3])
    u1 = Conv2D(256, 3, activation='relu', padding='same')(u1)
    
    u2 = UpSampling2D()(u1)
    u2 = Concatenate()([u2, c2])
    u2 = Conv2D(128, 3, activation='relu', padding='same')(u2)
    
    u3 = UpSampling2D()(u2)
    u3 = Concatenate()([u3, c1])
    u3 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(u3)
    
    return Model(inputs, outputs)

# Compile model
model = unet()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
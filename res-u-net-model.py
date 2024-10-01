from keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras.models import Model

def nested_unet(input_shape):
    inputs = Input(input_shape)
    # Define U-Net++ architecture here...
    # For brevity, only the basic structure is shown
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # Continue defining the layers...
    model = Model(inputs, conv1)  # Replace with the actual output layer
    return model

from keras.layers import multiply

def attention_unet(input_shape):
    inputs = Input(input_shape)
    # Define Attention U-Net architecture here...
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # Implement attention mechanism...
    model = Model(inputs, conv1)  # Replace with the actual output layer
    return model


from keras.losses import binary_crossentropy
from keras.metrics import MeanIoU

# DICE score function
def dice_coefficient(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

# Compile models
model_nested = nested_unet((256, 256, 1))
model_attention = attention_unet((256, 256, 1))

model_nested.compile(optimizer='adam', loss=binary_crossentropy, metrics=[dice_coefficient])
model_attention.compile(optimizer='adam', loss=binary_crossentropy, metrics=[dice_coefficient])

# Train models
history_nested = model_nested.fit(datagen.flow(X_train, y_train), epochs=50, validation_data=(X_test, y_test))
history_attention = model_attention.fit(datagen.flow(X_train, y_train), epochs=50, validation_data=(X_test, y_test))

# Evaluate models
nested_score = model_nested.evaluate(X_test, y_test)
attention_score = model_attention.evaluate(X_test, y_test)




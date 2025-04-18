import tensorflow as tf
import numpy as np
from utils import get_model,load_train_and_test_data, data_preprocessing
EPOCHS=10
IMG_WIDTH,IMG_HEIGHT=31,31

def initialize_with_ResNet(regularizer_strength,dropout_rate, img_width, img_height):
    ResNet_copy=tf.keras.applications.ResNet50(weights='imagenet')
    for layer in ResNet_copy.layers:
        if 'conv1_conv' in layer.name:
            print(f"Layer name: {layer.name} is obtained.")
            weights,bias=layer.get_weights()
            break
    model=get_model(regularizer_strength,dropout_rate,True, img_width, img_height)
    new_model = tf.keras.models.Sequential()
    for i, layer in enumerate(model.layers):
        if i == 0: 
            new_model.add(tf.keras.layers.Conv2D(64, (7, 7), activation="relu", padding="same", kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),)
        elif i==1:
            new_model.add(layer)
        else:
            break
    new_model.layers[0].set_weights([weights, bias])
    return new_model

def tranfer_learning():
    # load the train and test data
    x_train, y_train, x_test, y_test = load_train_and_test_data("data", IMG_WIDTH, IMG_HEIGHT)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    x_train, x_test = data_preprocessing(x_train, x_test)

    # one hot encoding
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # a possible value of dropout_rates and regularizer_strengths for overfitting
    dropout_rate=0.1
    regularizer_strength=0.3

    model = get_model(dropout_rate,regularizer_strength, batch_normalization=True, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2, batch_size=32)
    

if __name__ == "__main__":
    tranfer_learning()
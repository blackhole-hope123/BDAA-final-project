import tensorflow as tf
import numpy as np
import pandas as pd
from model_training import get_model
from data_preprocessing import data_preprocessing
import data_loading
import matplotlib.pyplot as plt

EPOCHS = 10
num_of_categories=43
IMG_WIDTH=30
IMG_HEIGHT=30

def get_model(regularizer_strength,dropout_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength), kernel_initializer='he_normal'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(num_of_categories, activation="softmax")
    ])
    model.compile(
        optimizer="nadam",  
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model trained.")
    return model
    
# load the train and test data
x_train, y_train, x_test, y_test = data_loading.x_train, data_loading.y_train, data_loading.x_test, data_loading.y_test  
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
# one hot encoding
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
print("Data preprocessing completed.")

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#test the best setup for dropout_rates and regularizer_strengths
dropout_rates=[i/10 for i in range (1,8)]
regularizer_strengths=[i/10 for i in range (1,7)]
accuracies=[]

for dropout_rate in dropout_rates:
    for regularizer_strength in regularizer_strengths:        
        model = get_model(dropout_rate,regularizer_strength)

        history=model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

        # Evaluate neural network performance
        loss,test_accuracy=model.evaluate(x_test, y_test, verbose=2, batch_size=32)
        last_training_accuracy = history.history['accuracy'][-1]
        accuracies.append([dropout_rate,regularizer_strength,test_accuracy-last_training_accuracy])
        
# draw the heatmap
accuracies = np.array(accuracies)

x_vals = np.unique(accuracies[:, 0])
y_vals = np.unique(accuracies[:, 1])

z_vals = accuracies[:, 2].reshape(len(y_vals), len(x_vals))

# Plot heatmap
plt.imshow(z_vals, 
           extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
           origin='lower', 
           aspect='auto',
           cmap='viridis') 

plt.colorbar(label='accu(drop,reg)')
plt.xlabel('dropout_rate')
plt.ylabel('regularizer_strength')
plt.title('accuracies')
plt.show()
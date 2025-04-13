import tensorflow as tf
import numpy as np
from utils import get_model, draw_heatmap, load_train_and_test_data, data_preprocessing
import sys
EPOCHS = 10

  
def main():
    # load the train and test data
    x_train, y_train, x_test, y_test = load_train_and_test_data("data")
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    x_train, x_test = data_preprocessing(x_train, x_test)

    # one hot encoding
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # a possible range of dropout_rates and regularizer_strengths for overfitting
    dropout_rate=0.6
    regularizer_strength=0.1

    model = get_model(dropout_rate,regularizer_strength, batch_normalization=True)

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, shuffle=False)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2, batch_size=32)
        
    # save the model to a file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

if __name__ == "__main__":
    main()
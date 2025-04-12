import tensorflow as tf
num_of_categories=43
IMG_WIDTH=30
IMG_HEIGHT=30
regularizer_strength=0.3
dropout_rate=0.3
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.5), kernel_initializer='he_normal'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength)),
        tf.keras.layers.Dropout(dropout_rate), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(regularizer_strength)),
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

import cv2
import os
import pandas as pd
import numpy as np
import tensorflow as tf


num_of_categories=43
regularizer_strength=0.3
dropout_rate=0.3
IMG_WIDTH=30
IMG_HEIGHT=30

def load_info():
    SignId_to_ShapeId,ClassId_to_SignId={},{}
    df=pd.read_csv("data/Meta.csv")
    df=df[["ClassId","ShapeId","SignId"]]
    df_filled = df.fillna("0")
    for row in df_filled.itertuples():
        ClassId_to_SignId[row.ClassId]=row.SignId
        if row.SignId not in SignId_to_ShapeId:
            SignId_to_ShapeId[row.SignId]=row.ShapeId
        else:
            if SignId_to_ShapeId[row.SignId]!=row.ShapeId:
                print("SignId to ShapeId is not a function!")
    print("SignId to ShapeId is a function!")
    return (SignId_to_ShapeId,ClassId_to_SignId)

SignId_to_ShapeId,ClassId_to_SignId=load_info()

# a method suitable for both train data and test data loading

def load_data(data_dir, csv_file_path):
    images,labels=[],[]
    df=pd.read_csv(csv_file_path)
    df=df[["ClassId","Path"]]
    for row in df.itertuples():
        image_path=os.path.join(data_dir, row.Path)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        images.append(resized_image)
        labels.append(row.ClassId)
        '''print(images)
        print(type(images[0]))
        print(images[0].shape)
        print(labels)'''
    print("Data loaded")
    return (images,labels)

# loading the test and training data
def load_train_and_test_data(data_dir):
    train,test="Train.csv","Test.csv"
    files=set([f for f in os.listdir(data_dir)])
    if test in files:
        test_label_path=os.path.join(data_dir,test)
        x_train,y_train=load_data(data_dir,test_label_path)
    else:
        raise Exception("Test labels are not available")
    if train in files:
        train_label_path=os.path.join(data_dir,train)
        x_test,y_test=load_data(data_dir,train_label_path)
    else:
        raise Exception("Training labels are not available")
    return (x_train,y_train,x_test,y_test)


def data_preprocessing(x_train,x_test):
    # mean subtraction and normalization
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    print("Data preprocessing completed.")
    return (x_train,x_test)

EPOCHS = 10
num_of_categories=43
IMG_WIDTH=30
IMG_HEIGHT=30

# load the train and test data
x_train, y_train, x_test, y_test = load_train_and_test_data("data")
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
# one hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train, x_test=data_preprocessing(x_train, x_test)

def get_model(optimizer):
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
        optimizer=optimizer,  
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model trained.")
    return model


model = get_model("adam")

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

loss,accuracy = model.evaluate(x_test, y_test, verbose=2, batch_size=32)

print("the accuracy with nadam optimizer is ", accuracy)

# evaluate the nadam performance, expected to be close to the above
model = get_model("nadam")

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

loss,accuracy = model.evaluate(x_test, y_test, verbose=2, batch_size=32)

print("the accuracy with adam optimizer is ", accuracy)





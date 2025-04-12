import cv2
import os
import pandas as pd

IMG_WIDTH=30
IMG_HEIGHT=30

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
        labels.append(int(row.ClassId))
        '''print(images)
        print(type(images[0]))
        print(images[0].shape)
        print(labels)'''
    print("Data loaded from ", csv_file_path)
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

'''
Another way of importing the training data
'''
'''
def load_training_data(data_dir):
    x_train, y_train = [], []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for i in range(len(folders)):
        label = int(folders[i])
        new_data_dir = os.path.join(data_dir, folders[i])
        files = [f for f in os.listdir(new_data_dir) if os.path.isfile(os.path.join(new_data_dir, f))]
        for j in range(len(files)):
            file_path = os.path.join(new_data_dir, files[j])
            image = cv2.imread(file_path)
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            x_train.append(resized_image)
            y_train.append(label)
    return (x_train, y_train)
'''

x_train,y_train,x_test,y_test=load_train_and_test_data("data")
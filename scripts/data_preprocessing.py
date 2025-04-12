import numpy as np

def data_preprocessing(x_train,x_test):
    # mean subtraction and normalization
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    print("Data preprocessing completed.")
    return (x_train,x_test)
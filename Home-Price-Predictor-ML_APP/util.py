import pickle
import json
import numpy as np
import tensorflow as tf

__locations = None
__data_columns = None
__model_LR = None
__model_NN = None
# __model_XG = None

def get_estimated_price(location,sqft,bhk,bath, model):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    print([x])

    if model == 'NN':
        x = np.expand_dims(x, axis=0)
        pred = __model_NN.predict([x])[0].tolist()
        pred = round(pred[0], 2)
    # elif model == 'XG':
    #     pred = round(__model_XG.predict([x])[0],2)
    else:
        pred = round(__model_LR.predict([x])[0], 2)

    return pred


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./model/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model_LR
    global __model_NN
    # global __model_XG

    if __model_LR is None:
        with open('./model/trained model/banglore_home_prices_model.pickle', 'rb') as f:
            __model_LR = pickle.load(f)

    if __model_NN is None:
        __model_NN = tf.keras.models.load_model('./model/trained model/banglore_home_prices_model_NN.h5')

    # Having problem because of OpenCv and Macos (Maybe can run in Windows and with installed OpenCV)
    # if __model_XG is None:
    #     with open('./model/trained model/banglore_home_prices_model_xg.pickle', 'rb') as f:
    #         __model_XG = pickle.load(f)

    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location

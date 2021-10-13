from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer


class Network():
    def __init__(self, load):
        if(load):
            self.load()
        else:
            self.model_create()

    def model_create(self):
        self.model = Sequential()
        
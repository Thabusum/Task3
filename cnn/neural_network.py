from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the model
        model = Sequential()
        
        # Add first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(depth, height, width), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add second set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add third set of CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the network and add fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(total_classes, activation='softmax'))

        # Load the saved weights if they are provided
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        
        return model

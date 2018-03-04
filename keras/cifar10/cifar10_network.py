#Import modules
import keras
from keras.datasets import cifar10 #Image Dataset (50000 train + 10000 test data)
from keras.models import Sequential #Neural Network Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import rmsprop

#Import dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(x = input, y = output)

print("x_train shape: ",x_train.shape) #Show channel size (50000, 32, 32, 3)

y_train = keras.utils.to_categorical(y_train, 10) #Converts outputs to onehot encoding
y_test = keras.utils.to_categorical(y_test, 10) #Converts outputs to onehot encoding

# Squice values between 0 and 1 --> normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Create the network model

#Input Layer
model = Sequential()
model.add(Conv2D(filters = 32,
                 kernel_size = (3, 3),
                 input_shape = x_train.shape[1:],
                 activation = 'relu'))
#Hidden Layers
model.add(Conv2D(filters = 32,
                 kernel_size = (3, 3),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64,
                 kernel_size = (3, 3),
                 activation = 'relu'))
model.add(Conv2D(filters = 64,
                 kernel_size = (3, 3),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))

#Output Layer
model.add(Dense(10, activation='softmax'))

#Train the network
model.compile(optimizer = rmsprop(lr = 1e-4, decay = 1e-6),
              loss = "categorical_crossentropy",
              metrics = ['accuracy'])

model.fit(x_train, y_train,
         batch_size = 128,
         epochs = 1,
         verbose = 1,
         shuffle = True,
         validation_data=(x_test, y_test))

#Save the trained network for future
#Saving design of the model
model.save('cifar10_cnn.h5',
           include_optimizer = rmsprop(lr = 1e-4, decay = 1e-6),
           overwrite = True)

#Saving weights of model
model.save_weights('cifar10_weights.h5',
                   overwrite = True)

score = model.evaluate(x_test, y_test, verbose = 1)
print("Loss:",score[0])
print("Accuracy:",score[1])
print("Finished")

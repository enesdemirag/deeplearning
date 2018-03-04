#Import modules
from keras.datasets import cifar10 #Neural Network Model
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(x = input, y = output)

# Squice values between 0 and 1 --> normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.load_model('cifar10_cnn.h5') #Load trained network model
model.load_weights('cifar10_weights.h5') #Load trained network's weights

score = model.evaluate(x_test, y_test, verbose = 1)
print("Loss:",score[0])
print("Accuracy:",score[1])
print("Finished")

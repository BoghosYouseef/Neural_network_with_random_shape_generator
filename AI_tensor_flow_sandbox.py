import tensorflow as tf
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# # print(tf.__version__)
# #
# #
# #
# # mnist = tf.keras.datasets.mnist
# #
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # x_train = tf.keras.utils.normalize(x_train, axis= 1)
# # x_test = tf.keras.utils.normalize(x_test, axis= 1)
# #
# # # print(x_train, type(x_train), np.size(x_train), "x TRAIN")
# # # print(x_test, type(x_test), np.size(x_test), "x TEST")
# # # print(y_train, type(y_train), np.size(y_train), "y TRAIN")
# # # print(y_test, type(y_test), np.size(y_test), "y TEST")
# # #
# # # arr = [[[2341,2,3],[4,4,4],[9,9,9]]]
# # # print(np.size(arr))
# # data = mnist.load_data()
# #
# # for i in data:
# #     print(i)
# #     print(data,"+++++++++++",data[i])
# # model = tf.keras.models.Sequential()
# #
# # model.add(tf.keras.layers.Flatten())
# #
# # model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
# # model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
# # model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))
# #
# # model.compile(optimizer="adam",                                                             #60000 figures
#                                                                                               # array containts 47040000
#                                                                                               #
# #               loss="sparse_categorical_crossentropy",
# #               metrics=["accuracy"])
# #
# # model.fit(x_train, y_train, epochs= 3, verbose=1)
# #
# # val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2)               # 64 neurons : - 1s 64us/sample - loss: 0.0607 - accuracy: 0.9656
# #
# # model.save("num_reader.model")                                              # 128 neurons : - 1s 67us/sample - loss: 0.0468 - accuracy: 0.9725
# #
# # new_model = tf.keras.models.load_model("num_reader.model")                  # 256 neurons : - 1s 75us/sample - loss: 0.0414 - accuracy: 0.9768
# #
# # predictions = new_model.predict([x_test])                                   # 512 neurons : - 1s 83us/sample - loss: 0.0504 - accuracy: 0.9723
# #
# # for i in range(25):                                                         # 1024 neurons : - 1s 117us/sample - loss: 0.0451 - accuracy: 0.9745
# #      print(np.argmax(predictions[i]))
# #      plt.imshow(x_test[i], cmap= plt.cm.binary)
# #      plt.show()
# #      #print(f"YOU ARE NOW HERE {i}")
# #      #print(x_train[0])

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import keras
import tensorflow
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
np.set_printoptions(threshold=sys.maxsize)


DATADIR = "D:/PyCharm_Projects/Third_Semester_Programming/Final_program/Data_base"

# FOR DATA FROM Final_Program_v2.2
# TrainingData = "D:/PyCharm_Projects/Third_Semester_Programming/Final_program/Data_base/TrainingData"
# TestData = "D:/PyCharm_Projects/Third_Semester_Programming/Final_program/Data_base/TestData"

TrainingData = "D:/PyCharm_Projects/Third_Semester_Programming/dataset2/Training"
TestData = "D:/PyCharm_Projects/Third_Semester_Programming/dataset2/Testing"
# CATEGORIES = ["SQUARE", "RECTANGLE", "EMPTY"]   # FOR DATA FROM Final_Program_v2.2
CATEGORIES = ["Triangle", "Quadrilateral", "Pentagon", "Hexagon",    # FOR DATA FROM shape_generator.py
              "Heptagon", "Octagon", "Nonagon", "Decagon"]
#counter = 0
# file = open("StandardOutputdataBase.txt", "w")


def data_processing(dir, output_file_name=False, verbose=1):

    traininglist = []
    imagelist = []


    counter = 0

    if output_file_name:

        file = open(f"{output_file_name}.txt", "w")

    for img in os.listdir(dir):  #listing all the images in the directory of the data base


        # if "RECTANGLE" in img:
        #     traininglist.append(CATEGORIES.index("RECTANGLE"))    # FOR DATA FROM Final_Program_v2.2
        # if "SQUARE" in img:
        #     traininglist.append(CATEGORIES.index("SQUARE"))
        # if "Empty" in img:
        #     traininglist.append(CATEGORIES.index("EMPTY"))

        if "Triangle" in img:                                       #FOR DATA FROM shape_generator.py
            traininglist.append(CATEGORIES.index("Triangle"))
        if "Quadrilateral" in img:
            traininglist.append(CATEGORIES.index("Quadrilateral"))
        if "Pentagon" in img:
            traininglist.append(CATEGORIES.index("Pentagon"))
        if "Hexagon" in img:
            traininglist.append(CATEGORIES.index("Hexagon"))
        if "Heptagon" in img:
            traininglist.append(CATEGORIES.index("Heptagon"))
        if "Octagon" in img:
            traininglist.append(CATEGORIES.index("Octagon"))
        if "Nonagon" in img:
            traininglist.append(CATEGORIES.index("Nonagon"))
        if "Decagon" in img:
            traininglist.append(CATEGORIES.index("Decagon"))


        img = os.path.join(dir, img) #creating path for each image

        if verbose == 1:
            print("Now Processing ", img)

        # img = load_img(img)    # using keras image processing function to load the image
        # img_array = img_to_array(img)  # using keras image processing function to turn image into array

        img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE )

        if verbose == 2:
            print(img_array, "Image is being turned into an array")

        # img_array *= (10/img_array.max())  # image normalization between 0 and 10
        new_array = cv2.resize(img_array, (100, 100))
        imagelist.append(new_array)
        #plt.imshow(new_array)
        #plt.gca().invert_yaxis()
        if verbose == 2:
            print(new_array)
            print("This is the new array with shape: ", new_array.shape)

        counter += 1

        if verbose:
            print("image number: ", counter, " image information: ", img)
        # print(f"{new_array} \n +++++++++++++++++++++++++++++ {counter}\n")
        # print(f"Shape: {new_array.shape}")
        #
        # plt.show()
        if output_file_name:
            file.write(f"{new_array} \n ############################################################# {counter}\n")

        # if counter == 1000:
        #     break
    if output_file_name:
        file.close()


    # print("This is the list of correct example catagories", traininglist)
    # print("############################################################################################################")
    # print(data)
    data = (np.array(imagelist, dtype=np.uint8), np.array(traininglist, dtype=np.uint8))

    return data


(x_train, y_train), (x_test, y_test) = data_processing(TrainingData, verbose=1), data_processing(TestData, verbose=1)


x_train = tf.keras.utils.normalize(x_train, axis= 1)
x_test = tf.keras.utils.normalize(x_test, axis= 1)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation= tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation= tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(8, activation= tf.nn.softmax))

model.compile(optimizer="adam",                                                             #53 figures

              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs= 50, verbose=1, shuffle=True)

val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2)               # 64 neurons : - 1s 64us/sample - loss: 0.0607 - accuracy: 0.9656

model.save("New_Shape_classifier.model")                                              # 128 neurons : - 1s 67us/sample - loss: 0.0468 - accuracy: 0.9725

random.shuffle(x_test)
random.shuffle(x_test)
random.shuffle(x_test)

new_model = tf.keras.models.load_model("New_Shape_classifier.model")                  # 256 neurons : - 1s 75us/sample - loss: 0.0414 - accuracy: 0.9768

predictions = new_model.predict([x_test])                                   # 512 neurons : - 1s 83us/sample - loss: 0.0504 - accuracy: 0.9723

for i in range(100):                                                         # 1024 neurons : - 1s 117us/sample - loss: 0.0451 - accuracy: 0.9745
    print(CATEGORIES[np.argmax(predictions[i])])
    plt.imshow(x_test[i], cmap="gray")
    plt.gca().invert_yaxis()
    plt.show()
    print(f"YOU ARE NOW HERE {i}")

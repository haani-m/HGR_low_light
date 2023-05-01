import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import cv2
import random
import numpy as np
import tensorflow
import keras
from random import shuffle
from keras.utils import np_utils
from shutil import unpack_archive

print("Imported Modules...")



class_dic = {"A":0,"B":1,"C":2, "D":3, "E":4, "F":5, "G":6, "H":7,"I":8, "J":9, "K":10,
             "L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,
          "U":20,"V":21,"W":22,"X":23,"Y":24, "Z":25}

print("Unpacked Dataset")

image_list = []
image_class = []

#path = r"C:\Users\Computing\OneDrive - University of Lincoln\Uni work\year 3\project\Dissertation\hgr_low_light\ASL_train"
#data_folder_path = r"C:\Users\Computing\OneDrive - University of Lincoln\Uni work\year 3\project\Dissertation\hgr_low_light\ASL_train"

path = "C:/Users/Computing/OneDrive - University of Lincoln/Uni work/year 3/project/Dissertation/hgr_low_light/ASL_train"
data_folder_path = "C:/Users/Computing/OneDrive - University of Lincoln/Uni work/year 3/project/Dissertation/hgr_low_light/ASL_train"

files = os.listdir(data_folder_path)



required_files = files
'''
symbols = []
for i in files:
	if i[0] in symbols:
		required_files.append(i)'''


for i in range(10):
	shuffle(files)

# print(type(files))
print(len(required_files))

class_count = {'A':0,'B':0,'C':0,"D":0,"E":0,"F":0,"G":0,"H":0,"I":0, "J":0, "K":0,
              "L":0,"M":0,"N":0,"O":0,"P":0,"Q":0,"R":0,"S":0,"T":0,
             "U":0,"V":0,"W":0,"X":0,"Y":0, "Z":0}

X = []
Y = []
X_val = []
Y_val = []
X_test = []
Y_test = []
unique_list=[] #Lists labels that have been trained on
train_num= 2400
test = 0
def preprocess(image):
    test = 0
    image = cv2.resize(image,(244,224))
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0, 135, 85])
    upper_skin = np.array([255, 180, 135])

    mask= cv2.inRange(ycbcr, lower_skin, upper_skin)

    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    procimage = cv2.bitwise_and(image, image, mask=binary)
    

    if test <51:
       cv2.imshow('result', procimage)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       test+=1
    
    return procimage


for file_name in required_files:
  label = file_name[0]

  if label not in unique_list:
    print(label)
    unique_list.append(label)

  path = data_folder_path+'/'+file_name
  image = cv2.imread(path)
  #preprocessed_image = preprocess(image)
  preprocessed_image = cv2.resize(image,(244,224))

  if class_count[label]<train_num:
    class_count[label]+=1
    X.append(preprocessed_image)
    Y.append(class_dic[label])

  #elif class_count[label]>=2000 and class_count[label]<2750:
  #  class_count[label]+=1
  #  X_val.append(preprocessed_image)
  #  Y_val.append(class_dic[label])

  else:
    X_test.append(preprocessed_image)
    Y_test.append(class_dic[label])

print(len(unique_list))
	
Y = np_utils.to_categorical(Y)
#Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

print(len(Y))
#print(len(Y_val))
print(len(Y_test))

print(len(X))
#print(len(X_val))
print(len(X_test))

npy_data_path  = "C:/Users/Computing/OneDrive - University of Lincoln/Uni work/year 3/project/Dissertation/hgr_low_light/Numpy"

np.save(npy_data_path+'/train_set.npy',X)
np.save(npy_data_path+'/train_classes.npy',Y)

#np.save(npy_data_path+'/validation_set.npy',X_val)
#np.save(npy_data_path+'/validation_classes.npy',Y_val)

np.save(npy_data_path+'/test_set.npy',X_test)
np.save(npy_data_path+'/test_classes.npy',Y_test)

print("Data pre-processing Success!")

#get train and validation sets
# npy_data_path  = "/content/drive/My Drive/ASL_Colab/Image_To_Numpy_Data"

X_train=np.load(npy_data_path+"/train_set.npy")
Y_train=np.load(npy_data_path+"/train_classes.npy")

#X_valid=np.load(npy_data_path+"/validation_set.npy")
#Y_valid=np.load(npy_data_path+"/validation_classes.npy")

X_test=np.load(npy_data_path+"/test_set.npy")
Y_test=np.load(npy_data_path+"/test_classes.npy")

X_test.shape

import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.applications import VGG16
from keras.preprocessing import image
from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D

print("Imported Network Essentials")

#Load the VGG model
image_size=224
vgg_base = VGG16(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

#initiate a model
model = Sequential()

#Add the VGG base model
model.add(vgg_base)

#Add new layers
model.add(Flatten())

model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))



# (4) Compile 
sgd = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint("Weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# (5) Train
model.fit(X_train/255.0, Y_train, batch_size=32, epochs=15, verbose=1,validation_data=(X_test/255.0,Y_test/255.0), shuffle=True,callbacks=[checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open("Model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Model/model_weights.h5")
print("Saved model to disk")



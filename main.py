import numpy as np
import os
import cv2
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


#################################

path="myData"
testRatio=0.2
imageDim=(32,32,3)
batchSize=50
epochsVal=10
stepsPerEpochs=300
#################################

images=[]
classNo=[]
className=[]
myList=os.listdir(path)
print(myList)
noOfClass=len(myList)

for x in range (0,noOfClass):
    myPicList=os.listdir(path+"/"+str(x))
    for y in myPicList:
        currImg=cv2.imread(path+"/"+str(x)+"/"+y)
        currImg=cv2.resize(currImg,(32,32))
        images.append(currImg)
        classNo.append(x)
    print(x,end=' ')
print(" ")

images=np.array(images)
classNo=np.array(classNo)


# splliting the data____________________

x_train,x_test,y_train,y_test=train_test_split(images,
                                               classNo,
                                               test_size=testRatio,
                                               random_state=1)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,
                                                           y_train,
                                                           test_size=testRatio,
                                                           random_state=1)
# print(x_train.shape())
# print(x_train.shape())

noOfSample=[]
for x in range(0,noOfClass):
    noOfSample.append(len(np.where(y_train==x)[0]))
print(noOfSample)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClass),noOfSample)
plt.title("No of Images for each classes")
plt.xlabel("Class Id")
plt.ylabel("No of images")
plt.show()


def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

x_train=np.array(list(map(preprocessing,x_train)))
x_test=np.array(list(map(preprocessing,x_test)))
x_validation=np.array(list(map(preprocessing,x_validation)))


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1],
                                    x_validation.shape[2], 1)


dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
dataGen.fit(x_train)

y_train=to_categorical(y_train,noOfClass)
y_test=to_categorical(y_test,noOfClass)
y_validation=to_categorical(y_validation,noOfClass)


def myModel():
    noOfFilters=60
    sizeOfFilters1=(5,5)
    sizeOfFilters2=(3,3)
    sizeOfPool=(2,2)
    noOfNodes=500

    model=Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilters1,
                      input_shape=(imageDim[0],imageDim[1],1),
                      activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilters1,  activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add((Conv2D(noOfFilters//2, sizeOfFilters2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClass,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model=myModel()
print(model.summary())

history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batchSize),
                            steps_per_epoch=stepsPerEpochs,epochs=epochsVal,
                    validation_data=(x_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('LOSS')
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('ACCURACY')
plt.xlabel("epoch")
plt.show()

score=model.evaluate(x_test,y_test,verbose=0)
print('Test Score-',score[0])
print('Test Accuracy-',score[1])



model.save("model_train.h5")















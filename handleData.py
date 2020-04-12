import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import glob
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


train = False

def switchOutput(pos):
    if(pos == 0):
        return 'Joao'
    elif(pos == 1):
        return 'Rita'
    elif(pos == 2):
        return 'Mae'
    elif(pos == 3):
        return 'Pai'
    else:
        return 'Not in label'


def imgResizeGrayScale(path):
    img = cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized_image = cv2.resize(gray_image, (50, 50))
    cv2.imwrite(path,resized_image)
    #print('img {path} resized'.format(path = path))

def addTodataset(fotopath, label, datasetname):
    df_path = f'./{datasetname}.pkl'
    imgResizeGrayScale(fotopath)
    img = cv2.imread(fotopath,0) # reads image 'opencv-logo.png' as grayscale
    if(train):
        if(os.path.exists(df_path)):
            olddf = pd.read_pickle(df_path)
            a_series = pd.Series([label, img], index = olddf.columns)
            newdf = olddf.append(a_series, ignore_index=True)
            newdf.to_pickle(df_path)
        else:
            df = pd.DataFrame(columns=['label', 'img'])
            a_series = pd.Series([label, img], index = df.columns)
            df = df.append(a_series, ignore_index=True)
            df.to_pickle(df_path)
        

def createModel(imgrow, imgcol):
    model = models.Sequential()
    model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(imgcol, imgrow, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def trainModel(datasetname):
    
    df_path = f'./{datasetname}.pkl'
    df = pd.read_pickle(df_path)
    # for i in range(0, len(df)):
    #     if(df['label'][i] == 0):
    #         wave = df['img'][i]
    #         plt.imshow(wave)
    #         plt.show()
    x_train = df['img'].values / 255.0
    y_train = df['label'].values
    imgcol = x_train[0].shape[0]
    imgrow = x_train[0].shape[1]
    temp = []
    for i in range(0, len(df)):
        temp.append(x_train[i])
    temp = np.asarray(temp, dtype=np.float64)
    x_train = temp.reshape((-1, imgrow, imgcol, 1))

    model = createModel(imgrow, imgcol)

    checkpoint_path = "training_weights/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period = 10)

    #Save the weights using the `checkpoint_path` format

    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])
    plotSetting(history, './saved.png')

def testModel(fotopath):
    imgResizeGrayScale(fotopath)
    imgarr1 = cv2.imread(fotopath,0)
    imgarr = imgarr1/ 255.0
    files = glob.glob('./faces/*')
    for f in files:
        os.remove(f)
    temp = []
    imgcol = imgarr.shape[0]
    imgrow = imgarr.shape[1]
    for i in range(0, len(imgarr)):
        temp.append(imgarr[i])
    temp = np.asarray(temp, dtype=np.float64)
    x_test = temp.reshape((-1, imgrow, imgcol, 1))
    checkpoint_path = "training_weights/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = createModel(imgrow, imgcol)
    model.load_weights(latest)
    predictions = model.predict(x_test)
    prediction = int(np.argmax(predictions[0]))
    percentage = round(float(predictions[0][prediction])*100,1)
  
    return switchOutput(prediction), percentage



def plotSetting(history, path):
    history_dict = history.history
    history_dict.keys()
    print(history_dict)
    acc = history_dict['acc']
    loss = history_dict['loss']

    epochs = range(1, len(acc) + 1)
    plt.figure
    # "-r^" is for solid red line with triangle markers.
    plt.plot(epochs, loss, '-r^', label='Training loss')
    # "-b0" is for solid blue line with circle markers.
    #plt.plot(epochs, val_loss, '-bo', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.figure
    plt.plot(epochs, acc, '-g^', label='Training acc')
    #plt.plot(epochs, val_acc, '-bo', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(path)
    plt.close()


# trainModel('trainpath')
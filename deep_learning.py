import os
import pandas as pd
import numpy as np
import librosa

from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation, MaxPooling1D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn






def main():
    path = "/Volumes/Transcend/BDA600/data_models/main_df.pkl"

    df = pd.read_pickle(path)

    emotion_map = {"happy": 0, "sad": 1, "angry": 2, "fearful": 5, "disgust": 4, "neutral": 3,
                   "surprised": 6, "calm": 7}
    mapped = [emotion_map[i] for i in df["emotion"]]
    df["class"] = mapped

    df = df[df["class"] <= 3].reset_index(drop=True)

    # count = 0
    # lst = []
    # for i in df["path"]:
    #     print(count / len(df))
    #     x, sr = librosa.load(i, res_type="kaiser_fast")
    #     mfccs = np.mean(librosa.feature.mfcc(y=x, sr = sr, n_mfcc=50).T, axis=0)
    #     mels = np.mean(librosa.feature.melspectrogram(y=x, sr=sr, n_mels=50).T, axis=0)
    #
    #     res = np.append(mfccs, mels)
    #     arr = res, df["class"][count]
    #     count += 1
    #     lst.append(arr)
    #
    #
    # X, y = zip(*lst)
    # X, y = np.asarray(X), np.asarray(y)
    #
    # print(X.shape, y.shape)
    #
    # pd.to_pickle(X,"/Volumes/Transcend/BDA600/data_models/deep_mfccs_x.pkl")
    # pd.to_pickle(y, "/Volumes/Transcend/BDA600/data_models/deep_mfccs_y.pkl")


    X = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/deep_mfccs_x.pkl")
    y = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/deep_mfccs_y.pkl")



    model = Sequential()
    model.add(Conv1D(100, 5, padding="same", input_shape=(100,1)))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # model.add(Conv1D(256, 5, padding='same', input_shape=(100, 1)))  # 1
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5, padding='same'))  # 2
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(MaxPooling1D(pool_size=(8)))
    # model.add(Conv1D(128, 5, padding='same'))  # 3
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same')) #4
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same')) #5
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(128, 5, padding='same'))  # 6
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(4))  # 7
    # model.add(Activation('softmax'))
    #
    # print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)


    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    print(x_traincnn.shape, x_testcnn.shape)

    model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

    cnn_history = model.fit(x_traincnn, y_train, batch_size=25, epochs=70, validation_data=(x_testcnn, y_test))

    yhat = model.predict(x_testcnn)
    res = []
    for i in yhat:
        res.append(np.where(i == max(i))[0][0])

    print(precision_recall_fscore_support(y_test, res, average="macro"))

    print(confusion_matrix(y_test, res, labels=[0, 1, 2, 3]))


    df_cm = pd.DataFrame(confusion_matrix(y_test, res, labels=[0, 1, 2, 3]),
                             index=["happy", "sad", "angry", "neutral"],
                             columns=["happy", "sad", "angry", "neutral"])
     #plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("1d CNN")
    plt.show()









if __name__ == "__main__":
    main()
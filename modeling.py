import pandas as pd
import numpy as np
from scipy import stats
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import librosa
#from featurewiz import featurewiz
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold

def cont_cont_heatmap(continuous, dataframe):
    result = []
    for i in continuous:
        holder = []
        for j in continuous:
            holder.append(
                np.round(stats.pearsonr(dataframe[i].values, dataframe[j].values)[0], 3)
            )
        result.append(holder)

    fig = ff.create_annotated_heatmap(
        result, x=continuous, y=continuous, showscale=True, colorscale="Blues"
    )
    fig.update_layout(title="Continuous-Continuous Correlation Matrix")
    fig.show()

def extract_mfcc(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def extract_mel(file_name):
    audio, sample_rate = librosa.load(file_name)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels = 50)
    mel_processed = np.mean(mel.T, axis=0)
    #mel_processed = np.log(mel_processed)

    return mel_processed

def extract_lpc(file_name):
    audio, sample_rate = librosa.load(file_name)
    lpc = librosa.lpc(audio,order=12)

    return lpc

def extract_rms(file_name):
    audio, sample_rate = librosa.load(file_name)
    rms = librosa.feature.rms(y=audio)
    rms_processed = np.mean(rms.T, axis=1)

    return rms_processed

def extract_zero(file_name):
    audio, sample_rate = librosa.load(file_name)
    zero = librosa.feature.zero_crossing_rate(y=audio)
    mean_zero = np.mean(zero)
    return mean_zero



# this function will take in a pretrained model, audio file, and fitted scaler to provide a
# prediction for a single audio file. Will be used in the app.py file

#PARAMETERS:
# Model: pretrained model to use to predict
# features: list of features in the original model
# audio_file: file to predict
# scaler: std or minmax scaler
def prediction_service(model, features, audio_file, scaler):
    # first need to extract all of the relevant features included in the trained model
    # (contained in "Classification_Models" directory
    mfcc_holder = extract_mfcc(audio_file)
    mel_holder = extract_mel(audio_file)
    lpc_holder = extract_lpc(audio_file)
    rms_holder = extract_rms(audio_file)
    zero_holder = extract_zero(audio_file)

    mfcc_df = pd.DataFrame([mfcc_holder])
    curr = 1
    mfcc_cols = []
    for i in range(len(mfcc_df.columns)):
        mfcc_cols.append("mfcc" + str(curr))
        curr += 1

    mfcc_df.columns = mfcc_cols

    mel_df = pd.DataFrame([mel_holder])
    curr = 1
    mel_cols = []
    for i in range(len(mel_df.columns)):
        mel_cols.append("mel" + str(curr))
        curr += 1

    mel_df.columns = mel_cols

    lpc_df = pd.DataFrame([lpc_holder])
    curr = 1
    lpc_cols = []
    for i in range(len(lpc_df.columns)):
        lpc_cols.append("lpc" + str(curr))
        curr += 1

    lpc_df.columns = lpc_cols

    rms_df = pd.DataFrame([rms_holder])
    curr = 1
    rms_cols = []
    for i in range(len(rms_df.columns)):
        rms_cols.append("rms" + str(curr))
        curr += 1

    rms_df.columns = rms_cols

    final_df = pd.concat([mfcc_df.reset_index(drop=True),mel_df.reset_index(drop=True),
                          lpc_df.reset_index(drop=True), rms_df.reset_index(drop=True)], axis=1)

    final_df["mean_zero"] = zero_holder

    final_df = final_df[features]

    # print("True Columns: ", final_df.columns.tolist())
    # print("Desired Order: ", features)
    #
    # print("Passing the Following into Scaler: ", final_df)



    # next, need to scale the new observation using the fitted scaler (transform)
    x = pd.DataFrame(scaler.transform(final_df))
    #print("result from scaler: ", x)

    #lastly, return the prediction using the model.
    yhat = model.predict(x)
    emotion_map = {"happy": 0, "sad": 1, "angry": 2,  "neutral": 3}
    # print("raw prediction: ", yhat)
    # print("emotion prediction: ", list(emotion_map.keys())[list(emotion_map.values()).index(yhat[0])])
    return list(emotion_map.keys())[list(emotion_map.values()).index(yhat[0])]


def main():

    df_path ="/Volumes/Transcend/BDA600/data_models/"
    main_df = pd.read_pickle(df_path + "main_df.pkl")
    # mfcc, mel, lpc, rms, zero = prediction_service(None, None, main_df["path"][0], None)
    # print("mfcc:", mfcc)
    # print("mel", mel)
    # print("lpc", lpc)
    # print("rms", rms)
    # print("zero", zero)

    emotion_map = {"happy":0, "sad":1, "angry":2, "fearful":3, "disgust":4, "neutral":5,
                   "surprised":6, "calm":7}
    mapped = [emotion_map[i] for i in main_df["emotion"]]
    main_df["emotion_mapped"] = mapped

    # testing to see if droppiong these labels creates better results
    #main_df = main_df[(main_df["emotion_mapped"] != 6) & (main_df["emotion_mapped"] != 7)]


    # create new dataframe which gets first 40 mfcc values in audio file
    # mfcc_holder = []
    # mel_holder = []
    # lpc_holder = []
    # rms_holder = []
    #count = 0
    #for i in main_df["path"]:
    #    print(count/len(main_df)*100)
    #     mfcc_holder.append(extract_mfcc(i).tolist())
    #    mel_holder.append(extract_mel(i).tolist())
    #     lpc_holder.append(extract_lpc(i).tolist())
    #     #rms_holder.append(extract_rms(i).tolist())
    #     #print(zero_holder)
    #    count += 1
    #
    # # create this datafraem
    # mfcc_df = pd.DataFrame(mfcc_holder)
    # pd.to_pickle(mfcc_df, "/Volumes/Transcend/BDA600/data_models/mfcc_df")
    #
    # # rename the columns
    # column_names = []
    # curr = 1
    # for i in range(len(mfcc_df.columns)):
    #     column_names.append("mfcc" + str(curr))
    #     curr += 1
    #
    #
    # mfcc_df.columns = column_names
    # #mfcc_df["zero_crossing"] = zero_holder
    # mfcc_df["class"] = main_df["emotion_mapped"].values
    # pd.to_pickle(mfcc_df, "/Volumes/Transcend/BDA600/data_models/mfcc_df")
    #
    mfcc_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/mfcc_df")
    mfcc_df["mean_zero"] = main_df["mean_zero"].values
    # mel_df = pd.DataFrame(mel_holder)
    # cols = []
    # cont = 1
    # for i in range(len(mel_df.columns)):
    #     cols.append("mel" + str(cont))
    #     cont += 1
    # mel_df.columns = cols
    # pd.to_pickle(mel_df,"/Volumes/Transcend/BDA600/data_models/mel_df" )
    #
    # lpc_df = pd.DataFrame(lpc_holder)
    # cols = []
    # cont = 1
    # for i in range(len(lpc_df.columns)):
    #     cols.append("lpc" + str(cont))
    #     cont += 1
    #
    # lpc_df.columns = cols
    # pd.to_pickle(lpc_df,"/Volumes/Transcend/BDA600/data_models/lpc_df"  )

    # rms_df = pd.DataFrame(rms_holder)
    # cols = []
    # cont = 1
    # for i in range(len(rms_df.columns)):
    #     cols.append("rms" + str(cont))
    #     cont += 1
    #
    # rms_df.columns = cols
    # pd.to_pickle(rms_df,"/Volumes/Transcend/BDA600/data_models/rms_df")

    lpc_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/lpc_df")
    mel_df = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/mel_df")

    mfcc_df = pd.concat([mfcc_df.reset_index(drop=True),mel_df.reset_index(drop=True), lpc_df.reset_index(drop=True)], axis=1)
    mfcc_df = mfcc_df.reset_index(drop=True)
    #print(mfcc_df.head().to_string())

    mfcc_df = mfcc_df[(mfcc_df["class"] != 6) & (mfcc_df["class"] != 7)& (mfcc_df["class"] != 4)& (mfcc_df["class"] != 3)]
    class_res = []
    for i in mfcc_df["class"]:
        if i == 5:
            class_res.append(3)
        else:
            class_res.append(i)
    mfcc_df["class"] = class_res
    print("class counts: ", mfcc_df["class"].value_counts())
    #print(mfcc_df.head().to_string())
    print(len(mfcc_df))
    print(mfcc_df.columns)





    #feature_df = main_df[main_df.columns[3:]]
   # x_features = feature_df[feature_df.columns[:-1]]
    feature_list = [x for x in mfcc_df.columns if x != "class"]

    #print(feature_list[:70])
    x_features = mfcc_df[feature_list]
    #x_features = x_features[[x for x in x_features.columns[:14]]]
    x_features["mean_zero"] = mfcc_df["mean_zero"].values
    subset = list(x_features.columns[:14])
    subset2 = list(x_features.columns[40:61])
    subset3 = list(x_features.columns[91:])
    x_features = x_features[subset+subset2+subset3]
    #print("columns: ", x_features.columns)
    full_df = x_features
    full_df["target"] = mfcc_df["class"]

    print(x_features.columns)


    #features, data = featurewiz(full_df, target="target", feature_engg="", verbose=0)
    features = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl")
    data = full_df[features]
    data["target"] = full_df["target"]

    #pd.to_pickle(data, "/Volumes/Transcend/BDA600/data_models/feature_engineered_df")

    #data = pd.read_pickle("/Volumes/Transcend/BDA600/data_models/feature_engineered_df")
    drop_cols = []
    for i in data.columns:
        if data[i].isna().sum() > 0:
            print(i + " has null values, dropping...")
            drop_cols.append(i)

    data = data.drop(drop_cols, axis=1)



    #x_features = full_df[full_df.columns[:-1]]
    #print("columns: ", x_features.columns)
    # print(x_features.shape)
    #x_features = x_features.dropna()
    #data = data.dropna()


    x_features = data.loc[:, data.columns != "target"]
    print(x_features.columns.tolist())
    print(features)
    #x_features = x_features.dropna()
    #data = data.dropna()

    #print(x_features.head().to_string())


    #col = [x for x in x_features.columns]
    # plot correlation matrix
    #cont_cont_heatmap(col, x_features)

    # get target variable
    #target = main_df["emotion_mapped"]
    target = data["target"]

    #print(x_features)

    # create a train test split
    x_train, x_test, y_train, y_test = train_test_split(x_features, target.values, test_size=0.2, random_state=42)

    #print("x_train: ", x_train.loc[0])
    #print("x_features: ", x_features)

    # standardize x_features using standard scaler
    scaler = StandardScaler()
    scaler_min = MinMaxScaler()
    scaler.fit(x_train)
    scaler_min.fit(x_train)

    print("passing following into scaler: ", x_train)

    x_train_std = pd.DataFrame(scaler.transform(x_train))
    x_train_min = pd.DataFrame(scaler_min.transform(x_train))
    print("scaler result: ", x_train_std)

    x_test_std = pd.DataFrame(scaler.transform(x_test))
    x_test_min = pd.DataFrame(scaler_min.transform(x_test))

    #print(x_features.values)


    # implement 5 fold cross validation
    kf = KFold(n_splits=5)

    tree = DecisionTreeClassifier()
    svc = SVC(kernel = 'linear')
    forest = RandomForestClassifier(n_jobs=-1)
    nn = MLPClassifier()

    acc_score = []
    acc_score_svc = []
    acc_score_forest = []
    acc_score_nn = []
    forest_params = {"n_estimators":[100,200,300],
                     "criterion":["genie","entropy"],
                     "max_depth":[None, 10, 50],
                     "bootstrap":[True, False]
                     }
    forest_search = RandomizedSearchCV(forest, param_distributions=forest_params, cv=5, n_jobs=-1,
                                       n_iter=70, random_state=100)
    forest_search.fit(x_train_std, y_train)
    best_random = forest_search.best_estimator_
    yhat = best_random.predict(x_test_std)

    print(y_test)
    print(yhat)

    print("Accuracy: ", accuracy_score(y_test, yhat))
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == yhat[i]:
            count += 1

    print(count/len(y_test))


    #writing model and list of selected features to a pickle file in "Classification_Models directory
    pd.to_pickle(best_random, "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/random_forest.pkl")
    pd.to_pickle(features,"/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl" )
    best_random = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/random_forest.pkl")
    features = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl")
    pd.to_pickle(scaler,"/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/scaler.pkl")





    # for train_index, test_index in kf.split(x_features_scaled):
    #     curr_X_train, curr_X_test = x_features_scaled[train_index, :], x_features_scaled[test_index, :]
    #     curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
    #
    #     print("fitting tree")
    #     tree.fit(curr_X_train, curr_y_train)
    #     print("fitting NN")
    #     nn.fit(curr_X_train, curr_y_train)
    #     print("fitting forest")
    #     forest.fit(curr_X_train, curr_y_train)
    #     pred_values = tree.predict(curr_X_test)
    #     pred_values_nn = nn.predict(curr_X_test)
    #     pred_values_forest = forest.predict(curr_X_test)
    #
    #     acc = accuracy_score(curr_y_test, pred_values)
    #     acc_nn = accuracy_score(curr_y_test, pred_values_nn)
    #     acc_forest = accuracy_score(curr_y_test, pred_values_forest)
    #
    #     acc_score.append(acc)
    #     acc_score_nn.append(acc_nn)
    #     acc_score_forest.append(acc_forest)
    #
    # print(acc_score)
    # print('NN', acc_score_nn)
    # print(acc_score_forest)

    # forest.fit(x_train_std, y_train)
    # print("fitting NN")
    # nn.fit(x_train_std, y_train)
    # feature_importances = forest.feature_importances_
    # importance = []
    # feature = []
    # for i in range(len(feature_importances)):
    #     #print(feature_importances[i], x_features.columns[i])
    #     feature.append(x_features.columns[i])
    #     importance.append(feature_importances[i])
    #
    # feature_dict = {"Features":feature, "Importance":importance}
    # feature_importance_df = pd.DataFrame(feature_dict)
    # print(feature_importance_df.sort_values(by="Importance", ascending=False))
    #
    # y_pred = forest.predict(x_test_std)
    # y_pred_nn = nn.predict(x_test_std)
    # print("forest test acc std: ", accuracy_score(y_test, y_pred))
    # print("NN test acc std: ", accuracy_score(y_test, y_pred_nn))
    #
    # forest.fit(x_train_min, y_train)
    # print("fitting NN min")
    # nn.fit(x_train_min, y_train)
    #
    # y_pred = forest.predict(x_test_min)
    # y_pred_nn = nn.predict(x_test_min)
    # print("forest test acc min: ", accuracy_score(y_test, y_pred))
    # print("NN test acc min: ", accuracy_score(y_test, y_pred_nn))





if __name__ == "__main__":
    main()

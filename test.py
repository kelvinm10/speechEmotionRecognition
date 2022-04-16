from modeling import prediction_service
import pandas as pd



df_path ="/Volumes/Transcend/BDA600/data_models/"
main_df = pd.read_pickle(df_path + "main_df.pkl")

main_df = main_df[(main_df["emotion"] == "happy")|(main_df["emotion"] == "sad")|
                   (main_df["emotion"] == "neutral")|(main_df["emotion"] == "angry")].reset_index(drop=True)

scaler = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/scaler.pkl")
model = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/random_forest.pkl")
features = pd.read_pickle("/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl")


count = 0
for i in range(len(main_df)):
    true = main_df["emotion"][i]
    pred = prediction_service(model, features, main_df["path"][i], scaler)
    print("True Emotion: ", true)
    print("Predicted: ", pred)

    if true == pred:
        count += 1

print(count / len(main_df))



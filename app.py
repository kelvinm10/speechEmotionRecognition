from flask import Flask, render_template, request, redirect
from modeling import prediction_service
import os
from io import BytesIO
import pickle
import requests

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    features_url = "https://github.com/kelvinm10/speechEmotionRecognition/blob/kelvinsBranch/Classification_Models/features.pkl?raw=true"
    model_url = "https://github.com/kelvinm10/speechEmotionRecognition/blob/kelvinsBranch/Classification_Models/random_forest.pkl?raw=true"
    scaler_url = "https://github.com/kelvinm10/speechEmotionRecognition/blob/kelvinsBranch/Classification_Models/scaler.pkl?raw=true"

    mfile_features = BytesIO(requests.get(features_url).content)
    mfile_model = BytesIO(requests.get(model_url).content)
    mfile_scaler = BytesIO(requests.get(scaler_url).content)

    features = pickle.load(mfile_features)
    model = pickle.load(mfile_model)
    scaler = pickle.load(mfile_scaler)


    # model = pd.read_pickle(
    #     "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/random_forest.pkl")
    # scaler = pd.read_pickle(
    #     "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/scaler.pkl")
    # features = pd.read_pickle(
    #     "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl")

    uploads_dir = os.path.join(app.instance_path, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        file.save(os.path.join(uploads_dir, file.filename))
        #saves to /Users/KelvinM/src/BDA600project/speechEmotionRecognition/instance/uploads/
        # need to test if this will work for anyone who wants to run this app
        if file.filename == "":
            return redirect(request.url)

        if file:
            # recognizer = sr.Recognizer()
            # audioFile = sr.AudioFile(file)
            # with audioFile as source:
            #     data = recognizer.record(source)
            # transcript = recognizer.recognize_google(data, key=None)
            #print(os.path.join(uploads_dir, file.filename))
            transcript = prediction_service(model, features, os.path.join(uploads_dir, file.filename), scaler)


    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

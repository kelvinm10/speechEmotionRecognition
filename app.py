from flask import Flask, render_template, request, redirect
from modeling import prediction_service
import pandas as pd
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    model = pd.read_pickle(
        "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/random_forest.pkl")
    scaler = pd.read_pickle(
        "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/scaler.pkl")
    features = pd.read_pickle(
        "/Users/KelvinM/src/BDA600project/speechEmotionRecognition/Classification_Models/features.pkl")

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

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import plotly.express as px


def create_dataframe(ravdess_path, data_path):
    ravdess_emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    crema_emotions = {
        "SAD": "sad",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
    }
    savee_emotions = {
        "a": "angry",
        "d": "disgust",
        "f": "fearful",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprised",
    }

    processed_data = []
    features_matrix = []
    # loop through ravdess dataset and add path, filename, and emotion to the list
    for i in os.listdir(ravdess_path):
        if not i.startswith("."):
            print(i)
            for j in os.listdir(ravdess_path + i):
                if not j.startswith("."):
                    emotion = ravdess_emotions[j.split("-")[2]]
                    processed_data.append([ravdess_path + i + "/" + j, j, emotion])
                    features_matrix.append(extract_features(ravdess_path + i + "/" + j))

    print("Done with Ravdess")
    # loop through crema, savee, and tess data and add path, filename, and emotion to the list
    for i in os.listdir(data_path):
        if i == "Ravdess" or i.startswith("."):
            continue
        for j in os.listdir(data_path + i):
            if not j.startswith("."):
                if i == "Crema":
                    emotion = crema_emotions[j.split("_")[2]]
                elif i == "Tess":
                    if j.split("_")[-1].split(".")[0] == "ps":
                        emotion = "surprised"
                    elif j.split("_")[-1].split(".")[0] == "fear":
                        emotion = "fearful"
                    else:
                        emotion = j.split("_")[-1].split(".")[0]
                else:
                    split = j.split("_")[-1]
                    if len(split) == 7:
                        emotion = savee_emotions[split[0]]
                    else:
                        emotion = savee_emotions[split[0] + split[1]]

                processed_data.append([data_path + i + "/" + j, j, emotion])
                features_matrix.append(extract_features(data_path + i + "/" + j))

    # turn list of all data into pandas dataframe
    processed_df = pd.DataFrame(processed_data, columns=["path", "filename", "emotion"])
    features_df = pd.DataFrame(
        columns=[
            "path",
            "mean_stft",
            # "median_stft",
            "mean_cqt",
            # "median_cqt",
            "mean_cens",
            # "median_cens",
            "mean_melspect",
            # "median_melspect",
            "mean_mfcc",
            # "median_mfcc",
            "mean_rms",
            # "median_rms",
            "mean_spec_cent",
            # "median_spec_cent",
            "mean_spec_band",
            # "median_spec_band",
            "mean_spec_cont",
            # "median_spec_cont",
            "mean_spec_flat",
            # "median_spec_flat",
            "mean_spec_roll",
            # "median_spec_roll",
            "mean_tonnetz",
            # "median_tonnetz",
            "mean_zero",
            # "median_zero",
        ]
    )
    features_df = features_df.append(
        pd.DataFrame(features_matrix, columns=features_df.columns)
    )
    print(len(processed_df))
    print(len(features_df))
    result_df = processed_df.merge(features_df, on="path")
    print(len(result_df))
    return result_df


def create_wave_graph(processed_df, list_of_emotions, num_of_samples):
    emotion_count = 0
    count = 0
    for i in processed_df["emotion"]:
        if i in list_of_emotions and emotion_count < num_of_samples:
            audio_array, sampling_array = librosa.load(processed_df["path"][count])
            emotion_count += 1

        else:
            count += 1
            continue
        librosa.display.waveshow(audio_array, sr=sampling_array)
        plt.title(processed_df["emotion"][count])
        plt.show()
        count += 1


# this function will extract all of the librosa features from a single file
def extract_features(filename):
    x, sr = librosa.load(filename)

    # extract chroma stft first, then get mean and median
    stft = librosa.feature.chroma_stft(y=x, sr=sr)
    mean_stft = np.mean(stft)
    # median_stft = np.median(stft)

    # extract chroma cqt mean and median
    cqt = librosa.feature.chroma_cqt(y=x, sr=sr)
    mean_cqt = np.mean(cqt)
    # median_cqt = np.median(cqt)

    # extract chroma cens (chroma energy normalized)
    cens = librosa.feature.chroma_cens(y=x, sr=sr)
    mean_cens = np.mean(cens)
    # median_cens = np.median(cens)

    # extract melspectogram
    melspect = librosa.feature.melspectrogram(y=x, sr=sr)
    mean_melspect = np.mean(melspect)
    # median_melspect = np.median(melspect)

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    mean_mfcc = np.mean(mfcc)
    # median_mfcc = np.median(mfcc)

    # extract rms
    rms = librosa.feature.rms(y=x)
    mean_rms = np.mean(rms)
    # median_rms = np.median(rms)

    # extract spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    mean_spec_cent = np.mean(spec_cent)
    # median_spec_cent = np.median(spec_cent)

    # extract spectral bandwith
    spec_band = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    mean_spec_band = np.mean(spec_band)
    # median_spec_band = np.median(spec_band)

    # extract spectral contrast
    spec_cont = librosa.feature.spectral_contrast(y=x, sr=sr)
    mean_spec_cont = np.mean(spec_cont)
    # median_spec_cont = np.median(spec_cont)

    # extract spectral flatness
    spec_flat = librosa.feature.spectral_flatness(y=x)
    mean_spec_flat = np.mean(spec_flat)
    # median_spec_flat = np.median(spec_flat)

    # extract spectral rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=x, sr=sr)
    mean_spec_roll = np.mean(spec_roll)
    # median_spec_roll = np.median(spec_roll)

    # extract tonnetz
    tonnetz = librosa.feature.tonnetz(y=x, sr=sr)
    mean_tonnetz = np.mean(tonnetz)
    # median_tonnetz = np.median(tonnetz)

    # extract zero crossing rate
    zero = librosa.feature.zero_crossing_rate(y=x)
    mean_zero = np.mean(zero)
    # median_zero = np.median(zero)

    # Now we have all the mean and median features for the single filename, store all information in a list
    return [
        filename,
        mean_stft,
        # median_stft,
        mean_cqt,
        # median_cqt,
        mean_cens,
        # median_cens,
        mean_melspect,
        # median_melspect,
        mean_mfcc,
        # median_mfcc,
        mean_rms,
        # median_rms,
        mean_spec_cent,
        # median_spec_cent,
        mean_spec_band,
        # median_spec_band,
        mean_spec_cont,
        # median_spec_cont,
        mean_spec_flat,
        # median_spec_flat,
        mean_spec_roll,
        # median_spec_roll,
        mean_tonnetz,
        # median_tonnetz,
        mean_zero,
        # median_zero,
    ]


def main():

    # ravdess_path = "/Volumes/Transcend/BDA600/data/Ravdess/audio_speech_actors_01-24/"  # pragma: allowlist secret
    # data_path = "/Volumes/Transcend/BDA600/data/"
    data_models_path = "/Volumes/Transcend/BDA600/data_models/"
    # main_df = create_dataframe(ravdess_path, data_path)
    main_df = pd.read_pickle(data_models_path + "main_df.pkl")

    print(main_df["emotion"].value_counts())

    # fig = px.histogram(processed_df, "emotion")
    # fig.show()

    create_wave_graph(main_df, [ "angry"], 20)


    print(main_df.head())


if __name__ == "__main__":
    main()

from flask import Flask, render_template, request, redirect
import speech_recognition as sr
import pyaudio

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)

    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

# chunk = 1024  # Record in chunks of 1024 samples
# sample_format = pyaudio.paInt16  # 16 bits per sample
# channels = 1
# fs = 44100  # Record at 44100 samples per second
# seconds = 3
# filename = "output.wav"


# def main():



    # chunk = 1024  # Record in chunks of 1024 samples
    # sample_format = pyaudio.paInt16  # 16 bits per sample
    # channels = 2
    # fs = 44100  # Record at 44100 samples per second
    # seconds = 3
    # filename = "output.wav"
    #
    # p = pyaudio.PyAudio()  # Create an interface to PortAudio
    #
    # print('Recording')
    #
    # stream = p.open(format=sample_format,
    #                 channels=channels,
    #                 rate=fs,
    #                 frames_per_buffer=chunk,
    #                 input=True)
    #
    # frames = []  # Initialize array to store frames
    #
    # # Store data in chunks for 3 seconds
    # for i in range(0, int(fs / chunk * seconds)):
    #     data = stream.read(chunk)
    #     frames.append(data)
    #
    # # Stop and close the stream
    # stream.stop_stream()
    # stream.close()
    # # Terminate the PortAudio interface
    # p.terminate()
    #
    # print('Finished recording')
    #
    # # Save the recorded data as a WAV file
    # wf = wave.open(filename, 'wb')
    # wf.setnchannels(channels)
    # wf.setsampwidth(p.get_sample_size(sample_format))
    # wf.setframerate(fs)
    # wf.writeframes(b''.join(frames))
    # wf.close()





# if __name__ == "__main__":
#     main()

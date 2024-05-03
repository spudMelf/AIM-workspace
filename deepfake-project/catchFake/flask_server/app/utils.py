import librosa
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def wav_to_melspectrogram(input_file, image_size=(224, 224)):
    y, sr = librosa.load(input_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)  # Adjust figsize and dpi for desired image size
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')  # Turn off axis labels

    output_file = input_file[:-3] + 'png'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    print(f"Saved mel spectrogram to {output_file} successfully")

    plt.close()

# yet another utility function I should move to a utils file
import glob
def clear_directory(dir):
    files = glob.glob(dir + '/*')
    for f in files:
        os.remove(f)
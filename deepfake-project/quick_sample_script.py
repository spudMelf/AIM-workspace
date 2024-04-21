import librosa
import librosa.display
import IPython.display as ipd
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def wav_to_melspectrogram(input_file, image_size=(224, 224)):
    # Load the audio file
    y, sr = librosa.load(input_file)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel spectrogram without axis
    plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)  # Adjust figsize and dpi for desired image size
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')  # Turn off axis labels

    # Save the plot as a PNG image
    output_file = input_file[:-3] + 'png'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    print(f"Saved mel spectrogram to {output_file} successfully")

    # Close the plot to free up memory
    plt.close()

PATH = "../ElevenLabs_2024-04-21T04_04_05_Chris_pre_s50_sb75_se0_b_m2.wav"
wav_to_melspectrogram(PATH)
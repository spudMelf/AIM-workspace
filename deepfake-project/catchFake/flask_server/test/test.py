import requests
import os

"""img_path = "/Users/eamon/Desktop/ElevenLabs_2024-04-21T04_04_05_Chris_pre_s50_sb75_se0_b_m2.png"
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(img_path, 'rb')})
print(resp.text)"""

#file_path = "/Users/eamon/Desktop/ElevenLabs_2024-04-21T04_04_05_Chris_pre_s50_sb75_se0_b_m2.wav"
#file_path = "/Users/eamon/Desktop/release_in_the_wild/16254.wav"
#file_path = r"C:/Users/apank/Downloads/release_in_the_wild/release_in_the_wild"
#file_path = "/Users/eamon/Desktop/Record (online-voice-recorder.com)2.mp3"

#file_path = "/Users/eamon/Desktop/fisher_1_ours.wav"
file_path = "/Users/eamon/Desktop/ElevenLabs_2024-04-21T04_04_05_Chris_pre_s50_sb75_se0_b_m2.wav"

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(file_path, 'rb')})
print(f"Old model prediction: {resp.text}")

resp = requests.post("http://localhost:5000/predict_w_noise",
                     files={"file": open(file_path, 'rb')})
print(f"New model prediction: {resp.text}")


import requests

img_path = "/Users/eamon/Desktop/AIM/deepfake-project/deepfake_mel_spectrograms/1015.png"
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(img_path, 'rb')})

print(resp.text)
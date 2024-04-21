from flask import Flask, jsonify, request
from predict import get_prediction
from os import path
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import os
from PIL import Image

from utils import allowed_file, wav_to_melspectrogram, clear_directory

UPLOAD_FOLDER = '/Users/eamon/Desktop/AIM/deepfake-project/flask_server/app/user_files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# there is definitely a better way to do this but I got lazy tbh so this will work 
# until someone smarter tells me how to fix it
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        src = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dst = os.path.join(app.config['UPLOAD_FOLDER'], "temp.wav")
        if str(file.filename[-3:]) == 'mp3':
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
            sound = AudioSegment.from_file(dst, format='wav')
            sound = sound.set_frame_rate(22050)
            sound.export(dst, format="wav")
        else:
            sound = AudioSegment.from_file(src, format='wav')
            sound = sound.set_frame_rate(22050)
            sound.export(dst, format="wav")

        wav_to_melspectrogram(dst)
        
        with open(app.config['UPLOAD_FOLDER'] + "/temp.png", 'rb') as image:
            img_bytes = image.read()
            class_id, class_name = get_prediction(image_bytes=img_bytes)
            clear_directory(app.config['UPLOAD_FOLDER'])
            return jsonify({'class_id': class_id, 'class_name': class_name})
        
        



            
            
                
            


        

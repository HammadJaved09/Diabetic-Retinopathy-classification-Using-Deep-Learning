from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from preprocessing import preprocess

app = Flask(__name__)

model = tf.keras.models.load_model('efficientnetb3_binary_best.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = './static/' + file.filename
            file.save(filepath)

            img = preprocess(filepath)
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            confidence = float(pred[0][0])  
            if confidence > 0.5:
                diagnosis = "The person likely has Diabetic Retinopathy."
                percentage = round(confidence * 100, 2)
            else:
                diagnosis = "The person likely does not have Diabetic Retinopathy."
                percentage = round((1 - confidence) * 100, 2)

            return render_template('index.html', 
                                   prediction=diagnosis,
                                   confidence=percentage,
                                   image=file.filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

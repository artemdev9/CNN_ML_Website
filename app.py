from flask import Flask, request, render_template
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image as keras_image

# Load your Keras model
model = load_model('./bdcm_convnn_v3_96.keras')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have an 'index.html' file in a 'templates' directory.

def preprocess_image(uploaded_file, target_size=(224, 224)):
    # Read the image through a file stream
    image_stream = uploaded_file.read()
    image_stream = np.fromstring(image_stream, np.uint8)
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    
    # Resize and normalize the image
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Preprocess and predict
            image = preprocess_image(file)
            image = np.expand_dims(image, axis=0)  # Expand dimensions to fit the model input
            prediction = model.predict(image)
            # Format the prediction in a more readable way if necessary
            # Example class labels
            class_labels = ['bicycle', 'cars', 'deer', 'mountains']  # replace with your actual class labels

            # Assuming 'prediction' is your model's prediction
            prediction = prediction
 
            # Convert to percentages
            prediction_percentages = [value * 100 for value in prediction[0]]

            # Find the index of the highest prediction percentage
            max_index = np.argmax(prediction_percentages)

            # Get the corresponding class label and percentage
            most_likely_class = class_labels[max_index]
            most_likely_percentage = prediction_percentages[max_index]

            return f"{most_likely_class}: {most_likely_percentage:.2f}%"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

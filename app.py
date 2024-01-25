from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)

# Load the model (consider using a safer method for production)
model = load_model('./bdcm_convnn_v3_96.keras')

@app.route('/')
def index():
    # Ensure you have an 'index.html' file in a 'templates' directory.
    return render_template('index.html')  

def preprocess_image(uploaded_file, target_size=(224, 224)):
    # Read the image through a file stream
    image_stream = uploaded_file.read()
    # Using np.frombuffer instead of np.fromstring (deprecated)
    image_stream = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    
    # Resize and normalize the image
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400

    try:
        image = preprocess_image(file)
        image = np.expand_dims(image, axis=0)  # Expand dimensions to fit the model input
        prediction = model.predict(image)
        
        # Define class labels
        class_labels = ['Bicycle', 'Cars', 'Deer', 'Mountains']
        max_index = np.argmax(prediction[0])
        
        prediction_str = class_labels[max_index]
        return jsonify({'prediction': prediction_str})
    except Exception as e:
        # Handle errors in processing or prediction
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    # Check if the file is an allowed type (for example, a PNG or JPEG image)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    # Turn off debug mode in production!
    app.run(debug=False,host="0.0.0.0", port=8080)


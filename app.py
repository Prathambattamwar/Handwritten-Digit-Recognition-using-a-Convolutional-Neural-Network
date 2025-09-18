import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image # Import the Pillow library

# 1. Load the Pre-trained Model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

print("Model Loaded Successfully! ✅")

# 2. Define the Prediction Function
def predict_digit(image):
    if image is None:
        return ""

    # --- FIX APPLIED HERE ---
    # The input 'image' is a NumPy array. We first convert it to a PIL Image
    # to use the easy resize function. The model needs a 28x28 grayscale image.
    pil_image = Image.fromarray(image).resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert the resized PIL image back to a NumPy array
    image = np.array(pil_image)
    
    # Manually invert the colors of the image array
    image = 255 - image

    # Reshape and normalize the image for the model
    # Now this will work because the image is the correct size (28x28)
    image = image.reshape(1, 28, 28, 1).astype('float32')
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(image)[0]

    # Create a dictionary of labels (digits) and their probabilities
    confidences = {str(i): float(prediction[i]) for i in range(10)}
    return confidences


# 3. Create and Launch the Gradio Interface
# We remove the width and height arguments as we now handle resizing in our function
input_component = gr.Image(
    image_mode='L',
    sources=["upload"], 
    label="Upload Your Digit Image"
)

output_component = gr.Label(num_top_classes=3, label="Predictions")

iface = gr.Interface(
    fn=predict_digit,
    inputs=input_component,
    outputs=output_component,
    title="Handwritten Digit Recognizer ✍️",
    description="Upload an image of a handwritten digit (0-9), and the AI will predict what it is."
)

# Launch the web application
iface.launch()
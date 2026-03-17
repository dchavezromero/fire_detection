import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('ffirenet_v1.keras')


def predict_fire(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = (img_dims / 127.5) - 1

    # Raw prediction from the model
    prediction = model.predict(img_preprocessed)
    raw_val = prediction[0][0]

    print(f"Raw Model Output: {raw_val:.4f}")

    # ALPHABETICAL LOGIC:
    # Index 0 ('f'ire) | Index 1 ('n'ofire)
    # If the value is close to 0, it's FIRE.
    # If the value is close to 1, it's NO FIRE.

    if raw_val < 0.5:
        label = "🔥 FIRE DETECTED"
        color = 'red'
        # Confidence is how close it is to 0
        score = (1 - raw_val) * 100
    else:
        label = "✅ NO FIRE"
        color = 'green'
        # Confidence is how close it is to 1
        score = raw_val * 100

    plt.imshow(img)
    plt.title(f"{label}\n({score:.2f}% Confident)")
    plt.axis('off')
    plt.show()

test_image_path = '/home/aadkoli/archive/Forest Fire Dataset/Testing/nofire_0336.jpg'
predict_fire(test_image_path)
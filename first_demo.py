import tensorflow as tf
import os
import matplotlib.pyplot as plt

fire_path = '/home/aadkoli/archive/Forest Fire Dataset/Training/fire'
no_fire_path = '/home/aadkoli/archive/Forest Fire Dataset/Training/nofire'

model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def get_prediction(folder, label):
    # first image
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    path = os.path.join(folder, files[0])

    # resize 224x224
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)

    # prep for mobilenet
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    # run on gpu
    print(f"checking {label}...")
    feat = model.predict(arr)
    return img, feat

f_img, f_feat = get_prediction(fire_path, "fire")
n_img, n_feat = get_prediction(no_fire_path, "no fire")

# results
plt.subplot(1, 2, 1)
plt.imshow(f_img)
plt.title("fire")

plt.subplot(1, 2, 2)
plt.imshow(n_img)
plt.title("no fire")

plt.show()
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Put your own path here
path = '/home/aadkoli/archive/Forest Fire Dataset/Training'

# split images into training and testing sets
train = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="training", seed=123, image_size=(224, 224)
)
test = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="validation", seed=123, image_size=(224, 224)
)

# resize pixel values to -1 to 1 for the ai
norm = layers.Rescaling(1./127.5, offset=-1)
train = train.map(lambda x, y: (norm(x), y))
test = test.map(lambda x, y: (norm(x), y))

# mobilenet
eyes = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
eyes.trainable = False

# fire detection
model = models.Sequential([
    eyes,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid') # 0 for fire, 1 for nofire (alphabetical)
])

# rules
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train, validation_data=test, epochs=10)

model.save('ffirenet_v1.keras')

plt.plot(model.history.history['accuracy'])
plt.show()
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
#from tensorflow.keras.preprocessing.image import load_img

base_dir = "/W4/data/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")

#print("Sample happy image:")
#plt.imshow(load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
#plt.show()
#
#print("\nSample sad image:")
#plt.imshow(load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
#plt.show()


# Load the first example of a happy face
#sample_image  = load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")

# Convert the image into its numpy array representation
#sample_array = img_to_array(sample_image)

#print(f"Each image has shape: {sample_array.shape}")

#print(f"The maximum pixel value used is: {np.max(sample_array)}")

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

# GRADED FUNCTION: image_generator
def image_generator():
    ### START CODE HERE

    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        '/W4/data/',
        target_size=(75, 75),
        batch_size=10,
        class_mode='binary')

    ### END CODE HERE

    return train_generator

gen = image_generator()

def train_happy_sad_model(train_generator):

    # Instantiate the callback
    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(75, 75, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer=optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    history = model.fit(x=gen,
                        epochs=15,
                        callbacks=[callbacks]
                       )

    ### END CODE HERE
    return history

hist = train_happy_sad_model(gen)

print("Your model reached the desired accuracy after {} epochs".format(len(hist.epoch)))

if not "accuracy" in hist.model.metrics_names:
    print("Use 'accuracy' as metric when compiling your model.")
else:
    print("The metric was correctly defined.")
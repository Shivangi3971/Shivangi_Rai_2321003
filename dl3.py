import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from kerastuner.tuners import Hyperband
from tensorflow.keras.optimizers import Adam


def preprocess(item):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.cast(item['image'], tf.float32) / 255.0
    label = item['label']
    return image, label





def build_model(hp):
    model = Sequential()
    # The input shape is correct for 32x32 RGB images
    model.add(Flatten(input_shape=(32, 32, 3)))  
    # Tune the number of units in the first Dense layer
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units_1, activation='relu'))
    # Add more layers and tune them similarly
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units_2, activation='relu'))
    # Output layer
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# Load SVHN dataset
svhn_train, svhn_test = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=False)

# Preprocess the data
svhn_train = svhn_train.map(preprocess).batch(64)
svhn_test = svhn_test.map(preprocess).batch(64)

# Instantiate the tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='hyperband',
    project_name='svhn_classification'
)

# Perform the hyperparameter search
tuner.search(svhn_train, 
             validation_data=svhn_test, 
             epochs=10,  # Adjust based on your requirements
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(svhn_train, epochs=50, validation_data=svhn_test)  # Adjust epochs and other parameters as needed

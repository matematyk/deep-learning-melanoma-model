import tensorflow as tf
from model import make_model
from utilities import get_dataset



TRAINING_FILENAMES = tf.io.gfile.glob("train*.tfrec")

TEST_FILENAMES = tf.io.gfile.glob("test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Test TFRecord Files:", len(TEST_FILENAMES))


#https://keras.io/examples/keras_recipes/tfrecord/
#dset = train_dataset.take(count=12)

print(TRAINING_FILENAMES)
print(TEST_FILENAMES)

train_dataset = get_dataset(TRAINING_FILENAMES)
test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

image_batch, label_batch = next(iter(train_dataset))

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "melanoma_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)


strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = make_model(lr_schedule)

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=test_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

import tensorflow as tf



AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = [256, 256]


TRAINING_FILENAMES = tf.io.gfile.glob("train*.tfrec")

TEST_FILENAMES = tf.io.gfile.glob("test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Test TFRecord Files:", len(TEST_FILENAMES))


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image

from functools import partial
#import tensorflow as tf
#print(tf.__version__)
#https://colab.research.google.com/gist/ravikyram/350c57a4facc1801f0021845c10288b1/untitled472.ipynb#scrollTo=1eGATycOlyhb

def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    print(dataset)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE) #todo, czy osobny watek
    dataset = dataset.batch(BATCH_SIZE)
    
    return dataset

#https://keras.io/examples/keras_recipes/tfrecord/
#dset = train_dataset.take(count=12)

print(TRAINING_FILENAMES)

train_dataset = get_dataset(TRAINING_FILENAMES)
test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

image_batch, label_batch = next(iter(train_dataset))


for n in range(25):
    print(image_batch[n].shape)
    print(label_batch[n])
       

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


def make_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(name="auc"),
    )

    return model
strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = make_model()

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=test_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

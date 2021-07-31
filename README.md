# Playground for melanoma-model

# res: https://keras.io/examples/keras_recipes/tfrecord/

## Installation dataset:

* Create kaggle.json
copy into ```/home/<user>/.kaggle```

* Downloading a dataset 106GB. 
```bash
kaggle competitions download -c siim-isic-melanoma-classification
```

## SIIM-ISIC Melanoma Classification

Url:
https://www.kaggle.com/c/siim-isic-melanoma-classification/data?select=test


## A problem with tensorflow version:

```python
import tensorflow as tf
print(tf.__version__)
```

https://colab.research.google.com/gist/ravikyram/350c57a4facc1801f0021845c10288b1/untitled472.ipynb#scrollTo=1eGATycOlyhb


## Keras Architectures with pre-trained weights:

https://www.tensorflow.org/api_docs/python/tf/keras/applications

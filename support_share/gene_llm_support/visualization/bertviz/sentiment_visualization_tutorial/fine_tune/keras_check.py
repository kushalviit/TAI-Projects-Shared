import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
model = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
model.summary()
# 3Fine-tune on IMDb movie reviews.
#classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
#classifier.predict(["What an amazing movie!", "A total waste of my time."])

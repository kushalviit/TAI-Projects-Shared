##############################################
# created on: 09/28/2023
# project: GeneLLM
#
#
##############################################
import argparse
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import pandas as pd
import yaml


CONST_FILE="constants.yaml"
with open(CONST_FILE, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            #print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

URL = parsed_yaml['constants']['URL']
TARFILE = parsed_yaml['constants']['TARFILE']
AUTOTUNE = tf.data.AUTOTUNE
DATASET_NAME= parsed_yaml['constants']['DATASET_NAME']
MODEL_NAME = parsed_yaml['constants']['MODEL_NAME']
MODEL_URL_DICT= parsed_yaml['constants']['MODEL_URL_DICT']
PREPROCESSOR_URL_DICT = parsed_yaml['constants']['PREPROCESSOR_URL_DICT']

def dataset_download():
    files=os.listdir(os.getcwd())
    dataset_dir = os.path.join(os.getcwd(), DATASET_NAME)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    if DATASET_NAME not in files:
       tf.keras.utils.get_file(TARFILE, URL,
                               untar=True, cache_dir=os.getcwd(),
                               cache_subdir='')
       dataset_dir = os.path.join(os.getcwd(), DATASET_NAME)
       os.remove(os.path.join(os.getcwd(),TARFILE))
       # remove unused folders to make it easier to load the data
       remove_dir = os.path.join(train_dir, 'unsup')
       shutil.rmtree(remove_dir)
    
    return dataset_dir, train_dir, test_dir



def prepare_data(batch_size, seed, validation_split,train_dir):
    train_ds = tf.keras.utils.text_dataset_from_directory(train_dir,
                                                             batch_size=batch_size,
                                                             validation_split=validation_split,
                                                             subset='training',
                                                             seed=seed)
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    val_ds = tf.keras.utils.text_dataset_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        validation_split=validation_split,
                                                        subset='validation',
                                                         seed=seed)
    
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds



def tf_hub_handler(bert_model_name):
    tfhub_handle_encoder = MODEL_URL_DICT[bert_model_name]
    tfhub_handle_preprocess = PREPROCESSOR_URL_DICT[bert_model_name]
    
    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
    
    return tfhub_handle_encoder, tfhub_handle_preprocess


def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

def initialize_optimizer_hyperparameters(args,train_ds):
    print(args.init_lr)
    print(args.epochs)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(0.1*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=args.init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type=args.optimizer_type)
    return loss, metrics, optimizer

def fine_tune_on_gpu(classifier_model,train_ds,val_ds,epochs):
    #gpus = tf.config.list_physical_devices('GPU')
    #if gpus:
    #   try:
    #      # Currently, memory growth needs to be the same across GPUs
    #      for gpu in gpus:
    #          tf.config.experimental.set_memory_growth(gpu, True)
    #      logical_gpus = tf.config.list_logical_devices('GPU')
    #      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #   except RuntimeError as e:
    #      # Memory growth must be set before GPUs have been initialized
    #      print(e)
    
    history = classifier_model.fit(x=train_ds,
                                   validation_data=val_ds,
                                   epochs=epochs)
    return history, classifier_model


def save_train_logs(history):
    hdf=pd.DataFrame.from_dict(history.history)
    hdf.to_csv('train_val_logs.csv',index=False)



def save_model(classifier_model):
    saved_model= './{}_bert'.format(DATASET_NAME.replace('/', '_'))
    classifier_model.save(saved_model,include_optimizer=False)



def main(args):
    tf.get_logger().setLevel(args.message_level.upper())
    
    #data set download and assign
    dataset_dir,train_dir,test_dir=dataset_download()
    train_ds,val_ds = prepare_data(args.batch_size, args.seed, args.validation_split, train_dir)
    
    #select base bert model and type
    tfhub_handle_encoder, tfhub_handle_preprocess = tf_hub_handler(MODEL_NAME)
    
    #build model initialize hyperparameters and compile
    classifier_model = build_classifier_model(tfhub_handle_preprocess,tfhub_handle_encoder)
    loss,metrics, optimizer = initialize_optimizer_hyperparameters(args,train_ds)
    classifier_model.compile(optimizer= optimizer, loss= loss, metrics =metrics)
    
    #fine tune model/train model
    print(f'Training model with {tfhub_handle_encoder}')
    history,classifier_model=fine_tune_on_gpu(classifier_model,train_ds,val_ds,args.epochs)
    
    # save logs and model
    save_train_logs(history)
    save_model(classifier_model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message-level",help="Message displays by Tensorflow- Error , Warnings etc", type=str,default='ERROR')
    parser.add_argument("--batch-size",help="Batch size for training", type= int,default=32)
    parser.add_argument("--seed",help="Seed value for randomizing dataset", type= int,default=42)
    parser.add_argument("--validation-split",help="Percentage split for training and validation", type= float, default=0.2)
    parser.add_argument("--epochs",help="Number of epochs for fine tuning", type=int, default=5)
    parser.add_argument("--init-lr",help="Initial learning rate", type=float, default=3e-5)
    parser.add_argument("--optimizer-type",help="Optimizer algorithm and type", type=str, default='adamw')
    args = parser.parse_args()
    main(args)


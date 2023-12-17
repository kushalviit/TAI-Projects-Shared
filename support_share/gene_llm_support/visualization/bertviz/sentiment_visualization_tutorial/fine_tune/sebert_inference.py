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



def prepare_data(batch_size, test_dir):
    test_ds = tf.keras.utils.text_dataset_from_directory(test_dir,
                                                         batch_size=batch_size)
    class_names=test_ds.class_names
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return test_ds,class_names


def review_data(test_ds,class_names):
    for text_batch, label_batch in test_ds.take(1):
        for i in range(3):
            print(f'Review:{text_batch.numpy()[i]}')
            label = label_batch.numpy()[i]
            print(f'label:{label} ({class_names[label]})')
    return

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


def load_model(saved_model):
    sebert_model = tf.keras.models.load_model(saved_model)
    sebert_model.summary()
    return sebert_model


def evaluate(classifier_model,test_ds):
     loss, accuracy = classifier_model.evaluate(test_ds)
     print(f'Loss: {loss}')
     print(f'Accuracy: {accuracy}')



def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


def main(args):
    tf.get_logger().setLevel(args.message_level.upper())
    
    #data set download and assign
    dataset_dir,train_dir,test_dir=dataset_download()
    test_ds,class_names= prepare_data(args.batch_size, test_dir)
    review_data(test_ds,class_names)
    
    #build model initialize hyperparameters and compile
    classifier_model = load_model(args.saved_model)
    if args.evaluation == True:
           loss,metrics, optimizer = initialize_optimizer_hyperparameters(args,test_ds)
           classifier_model.compile(optimizer= optimizer, loss= loss, metrics =metrics)
           #evaluation
           print(f'Evaluating trained model')
           evaluate(classifier_model, test_ds)
    
    examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
    ]
    
    results = tf.sigmoid(classifier_model(tf.constant(examples)))
    print_my_examples(examples,results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message-level",help="Message displays by Tensorflow- Error , Warnings etc", type=str,default='ERROR')
    parser.add_argument("--batch-size",help="Batch size for training", type= int,default=32)
    parser.add_argument("--saved-model",help="Location of saved model", type=str, default="aclImdb_bert")
    parser.add_argument("--epochs",help="Number of epochs for fine tuning", type=int, default=5)
    parser.add_argument("--init-lr",help="Initial learning rate", type=float, default=3e-5)
    parser.add_argument("--optimizer-type",help="Optimizer algorithm and type", type=str, default='adamw')
    parser.add_argument("--evaluation",help="Evaluation on test data", type= bool,default=True)
    args = parser.parse_args()
    main(args)


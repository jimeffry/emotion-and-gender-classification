"""
File: train_gender_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train gender classification model
"""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from utils.datasets import DataManager
from models.cnn import mini_XCEPTION
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data
import argparse
import numpy as np

def args():
    parser = argparse.ArgumentParser(description="train gender models")
    parser.add_argument('--batch_size',type=int,default=128,\
                        help="the number images to process once")
    parser.add_argument('--num_epochs',type=int,default=10000,\
                        help="the num  times to run ")
    parser.add_argument('--dataset_name',type=str,default='Aidence',\
                        help="which dataset to train")
    parser.add_argument('--graymode',type=bool,default=False,\
                        help="whether use gray image")
    parser.add_argument('--img_dir',type=str,default='../datasets/Aidence_imgs/',\
                        help="the directory to saved images ")
    parser.add_argument('--mode',type=str,default='gender',\
                        help="train the mode: gender or emotion or age")
    parser.add_argument('--patience',type=int,default=1000,\
                        help='used as to early_stop and lr_decay')
    parser.add_argument('--lr',type=float,default=0.01,\
                        help="learning rate")
    parser.add_argument('--load_model',type=str,default=None,\
                        help="load pretrained model")
    parser.add_argument('--val_ratio',type=float,default=0.1,\
                        help="how many images to val")
    parser.add_argument('--anno_file',type=str,default="./prepare_data/train.txt",\
                        help="train annotation file")
    return parser.parse_args()

def main():
    # parameters
    param = args()
    batch_size = param.batch_size
    num_epochs = param.num_epochs
    validation_split = param.val_ratio
    do_random_crop = False
    patience = param.patience
    dataset_name = param.dataset_name
    grayscale =param.graymode
    mode = param.mode
    anno_file = param.anno_file
    if mode == "gender":
        num_classes = 2
    elif mode == "age":
        num_classes = 101
    elif mode == "emotion":
        num_classes = 7
    else:
        num_classes = 5
    if grayscale:
        input_shape = (64, 64, 1)
    else:
        input_shape = (64, 64, 3)
    images_path = param.img_dir
    log_file_path = '../trained_models/%s_models/%s_model/raining.log' % (mode,dataset_name)
    trained_models_path = '../trained_models/%s_models/%s_model/%s_mini_XCEPTION' % (mode,dataset_name,mode)
    pretrained_model = param.load_model
    print("-------begin to load data------",input_shape)
    # data loader
    data_loader = DataManager(dataset_name,anno_file)
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
    print('Number of training samples:', len(train_keys))
    print('Number of validation samples:', len(val_keys))
    train_image_generator = ImageGenerator(ground_truth_data, batch_size,
                                    input_shape[:2],
                                    train_keys,
                                    path_prefix=images_path,
                                    grayscale=grayscale)
    val_image_generator = ImageGenerator(ground_truth_data, batch_size,
                                    input_shape[:2],
                                    val_keys,
                                    path_prefix=images_path,
                                    grayscale=grayscale)

    # model parameters/compilation
    if pretrained_model != None:
        model = load_model(pretrained_model,compile=False)
        print("pretrained model:",model.input_shape)
    else:
        model = mini_XCEPTION(input_shape, num_classes)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    # model callbacks
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1,patience=int(patience), verbose=1,min_lr=0.0000001)
    csv_logger = CSVLogger(log_file_path, append=False)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # training model
    print("-----begin to train model----")
    model.fit_generator(train_image_generator.flow(),
                        steps_per_epoch=int(np.ceil(len(train_keys) / batch_size)),
                        epochs=num_epochs, verbose=1,
                        callbacks=callbacks,
                        validation_data=val_image_generator.flow(),
                        validation_steps=int(np.ceil(len(val_keys) / batch_size)))

if __name__=='__main__':
    main()
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.utils.visualize_util import plot

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Classifying images with and without tomatoes.")

    parser.add_argument('--rescale', type=float, nargs='?', default=1.0/255,
                        help='Rescaling factor.')

    parser.add_argument('--train_sample_per_epoch', type=int, nargs='?', default=552,
                        help='Number of training samples used for training per epoch.')

    parser.add_argument('--test_sample_per_epoch', type=int, nargs='?', default=2192,
                        help='Number of training samples used for testing per epoch.')

    parser.add_argument('--img_width', type=int, nargs='?', default=600,
                        help='Width of images. The images will be resized.')

    parser.add_argument('--img_height', type=int, nargs='?', default=600,
                        help='Height of images. The images will be resized.')

    parser.add_argument('--epoch', type=int, nargs='?', default=50,
                        help='Number of training epoch.')

    parser.add_argument('--train_data', nargs='?', default='Train',
                        help='Train directory.')

    parser.add_argument('--test_data', nargs='?', default='Test',
                        help='Test directory.')

    parser.add_argument('--layer_size', nargs='?', default='32,32,64',
                        help='Size of each layer.')

    parser.add_argument('--dropout', type=float, nargs='?', default=0.5,
                        help='Dropout ratio.')

    parser.add_argument('--batch_size', type=int, nargs='?', default=32,
                        help='Batch size.')

    parser.add_argument('--l2', type=float, nargs='?', default=0.0001,
                        help='l2 regularization.')

    parser.add_argument('--metrics', nargs='?', default='accuracy,fmeasure,precision,recall',
                        help='Metric used to evaluate the method.')

    parser.add_argument('--loss', nargs='?', default='mse', #binary_crossentropy ...
                        help='Objective function.')

    parser.add_argument('--optimizer', nargs='?', default='sgd',    #rmsprop...
                        help='Optimization method.')

    parser.add_argument('--class_weight', type=float, nargs='?', default=19,
                        help='Weight of first class')

    parser.add_argument('--log_file', nargs='?', default='log.txt',
                        help='Log file for training loss and other metrics.')

    parser.add_argument('--plot', nargs='?', default='model.png',
                        help='Output plot file name.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_data_dir = args.train_data
    validation_data_dir = args.test_data

    try:
        layer_sizes = [int(x) for x in args.layer_size.split(',')]
    except Exception, e:
        raise e    

    # Model settings
    model = Sequential()
    model.add(Convolution2D(layer_sizes[0], 3, 3, input_shape=(3, args.img_width, args.img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if len(layer_sizes) > 1:
        for i in range (len(layer_sizes) - 1):
            model.add(Convolution2D(layer_sizes[i+1], 3, 3))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(1, W_regularizer=l2(args.l2), activity_regularizer=activity_l2(args.l2)))
    model.add(Activation('sigmoid'))

    metrics = args.metrics.split(',')

    model.compile(loss=args.loss,
                  optimizer=args.optimizer,
                  metrics=metrics)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=args.rescale,
            # rotation_range=90,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=args.rescale)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(args.img_width, args.img_height),
            batch_size=args.batch_size,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(args.img_width, args.img_height),
            batch_size=args.batch_size,
            class_mode='binary')

    # to deal with imbalanced dataset
    class_weight = {0:1,1:args.class_weight}
    fit = model.fit_generator(
            train_generator,
            samples_per_epoch=args.train_sample_per_epoch,
            nb_epoch=args.epoch,
            validation_data=validation_generator,
            nb_val_samples=args.test_sample_per_epoch,
            class_weight=class_weight)

    with open (args.log_file, "w") as f:
        f.write (fit.history)

    # save model plot
    plot(model, to_file=args.plot)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Classifying images with and without tomatoes.")

    parser.add_argument('--rescale', nargs='?', default=1.0/255,
                        help='Rescaling factor.')

    parser.add_argument('--train_sample_per_epoch', nargs='?', default=552,
                        help='Number of training samples used for training per epoch.')

    parser.add_argument('--test_sample_per_epoch', nargs='?', default=2192,
                        help='Number of training samples used for testing per epoch.')

    parser.add_argument('--epoch', nargs='?', default=20,
                        help='Number of training epoch.')

    parser.add_argument('--train_data', nargs='?', default='Train',
                        help='Train directory.')

    parser.add_argument('--test_data', nargs='?', default='Test',
                        help='Test directory.')

    parser.add_argument('--batch_size', nargs='?', default=32,
                        help='Batch size')

    parser.add_argument('--metrics', nargs='?', default='fbeta_score',
                        help='Batch size')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # dimensions of our images.
    img_width, img_height = 600, 600

    train_data_dir = args.train_data
    validation_data_dir = args.test_data

    # Model settings
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[args.metrics])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=args.rescale,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=args.rescale)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=args.batch_size,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=args.batch_size,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            samples_per_epoch=args.train_sample_per_epoch,
            nb_epoch=args.epoch,
            validation_data=validation_generator,
            nb_val_samples=args.test_sample_per_epoch)
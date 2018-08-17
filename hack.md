'''Trains a convnet using ResNet50 pre-trained net. Based partly on code from user fujisan on kaggle.com, accessed at: https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/versions
Also adapted
from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import sys, os, re, pickle, datetime, time
from os import listdir
from os.path import isfile, join
from collections import Counter
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras import applications, optimizers, backend as K
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.applications import ResNet50
import matplotlib.pyplot as plt

sys.setrecursionlimit(1000000)

seed = 1337
np.random.seed(seed)  # for reproducibility


# Runs code on GPU
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

def read_data(data):
    ''' Reads in data loaded from saved numpy array.'''
    x = data.files[0]
    x = data[x]
    y = data.files[1]
    y = data[y]
    return x, y

def class_counts_specifications(y):
    '''
    Makes pandas df for connecting numerical indices to flower categories, counts number of images within each category. Also returns list of class names.
    '''
    class_labels = np.unique(y)
    # flower_cats = {}
    # for i, name in enumerate(class_labels):
    #     flower_cats[i] = name

    flower_cat_counter = Counter(y)

    flower_count_df = pd.DataFrame.from_dict(flower_cat_counter, orient='index')
    flower_count_df = flower_count_df.rename(columns={0: 'species'})
    flower_count_df['count'] = list(flower_cat_counter.values())

    return flower_count_df, class_labels

def train_validation_split(x, y):
    '''
    Splits train and validation data and images. (Will also load test images, names from saved array).
    Input: saved numpy array, files/columns in that array
    Output: Train/validation data (e.g., X_train, X_test, y_train, y_test), test images, test image names (file names minus '.png')
    '''
    # Encode flower categories as numerical values
    number = LabelEncoder()
    y = number.fit_transform(y.astype('str'))

    # Split train and test subsets to get final text data (don't change this)
    X_training, X_test_holdout, y_training, y_test_holdout = train_test_split(x, y, stratify=y, random_state=42, test_size=.2)
    print('Initial split for (holdout) test data:\n \
    X_training: {} \n \
    y_training: {} \n \
    X_test_holdout: {} \n \
    y_test_holdout: {} \n'.format(X_training.shape, y_training.shape, X_test_holdout.shape, y_test_holdout.shape))

    # Split train into train and validation data (different for each model):
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, stratify=y_training, random_state=seed, test_size=.2)
    train_classes = len(np.unique(y_train))
    test_classes = len(np.unique(y_test))

    print('Train/validation split for this model:\n \
    X_train: {} \n \
    y_train: {} \n \
    X_test: {} \n \
    y_test: {} \n \
    n_train_classes: {} \n \
    n_test_classes: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape, train_classes, test_classes))

    # Standardize pixel values (between 0 and 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_test_holdout = X_test_holdout.astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    X_test_holdout = X_test_holdout/255

    return X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout

def convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes):
    ''' Converts class vectors to binary class matrices'''
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_test_holdout = np_utils.to_categorical(y_test_holdout, nb_classes)
    return Y_train, Y_test, Y_test_holdout

def build_cnn_resnet_50(input_shape=(224,224,3)):
    ''' Builds and compiles CNN with ResNet50 pre-trained model.
    Input: Shape of images to feed into top layers of model
    Output: Compiled model (final_model), summary of compiled model
    '''
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # add_model.add(Dropout(0.5))
    add_model.add(Dense(nb_classes, activation='softmax'))

    # Combine base model and my fully connected layers
    final_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # Specify SGD optimizer parameters
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return final_model, final_model.summary()

def _image_generator(X_train, Y_train):
    # seed = 135
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_datagen.fit(X_train, seed)
    return train_datagen

def fit_model_resnet50(X_train, X_test, Y_train, Y_test, save_output_root, model_type, name_time, batch_size, epochs, input_shape):
    print('\nBatch size: {} \nCompiling model...'.format(batch_size))
    generator = _image_generator(X_train, Y_train)

    # checkpoint
    filepath='weights/weights-improvement142-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    # Change learning rate when learning plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=2, min_lr=0.00001)

    # Stop model once it stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

    # put all callback functions in a list
    callbacks_list = [checkpoint, reduce_lr]

    history = final_model.fit_generator(
        generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=(X_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list, shuffle=True
        )

    score = final_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    ypred = final_model.predict(X_test)
    # ypred_classes = final_model.predict_classes(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred, final_model, history

# def visualize_layers(model):
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])

def model_summary_plots(history, save_output_root, model_type, name_time):
    print(history.history.keys())
    plt.close('all')
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}{}_{}/accuracy_plot'.format(save_output_root, model_type, name_time))
    plt.close('all')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('{}{}_{}/loss_plot'.format(save_output_root, model_type, name_time))

def sklearn_stats(Y_true, y_predicted, target_names):
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = y_test
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    return report

def predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout):
    test_predictions = model_fitted.predict(X_test_holdout)

    score = model_fitted.evaluate(X_test_holdout, Y_test_holdout, verbose=0, batch_size=batch_size)
    return test_predictions, score

def make_model_summary_file(name_time, save_output_root, model_type, start_time, finish_time, seed, input_shape, epochs, batch_size, nb_classes, X_train, X_test, X_test_holdout, score, flower_count_df, notes):
    with open('{}{}_{}/model_summary.txt'.format(save_output_root, model_type, name_time), 'a') as f:
        f.write('Model Summary \n \
        Start time: {} \n \
        Finish time: {} \n \
        Model: {} \n \
        Seed: {} \n \
        Input shape: {} \n \
        Epochs: {} \n \
        Batch size: {} \n \
        N classes: {} \n \
        X/Y_train size: {} \n \
        X/Y_test size: {} \n \
        X/Y_test_holdout: {} \n \n \
        Test score on X_test: {} \n \
        Accuracy score on X_test: {} \n \
        Flower categories: {} \n \n \
        Notes: {}'.format(start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, len(X_train), len(X_test), len(X_test_holdout), score[0], score[1], flower_count_df, notes))
        # F1 score: {} \n \
        # Precision: {} \n \
        # Recall: {} \n \
        # '

def save_model(name_time, history, model_fitted, flower_count_df, save_output_root, start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, X_train, Y_train, X_test, Y_test, X_test_holdout, Y_test_holdout, score):

    os.mkdir('{}{}_{}'.format(save_output_root, model_type, name_time))
    new_root = '{}{}_{}'.format(save_output_root, model_type, name_time)

    make_model_summary_file(name_time, save_output_root, model_type, start_time, finish_time, seed, input_shape, epochs, batch_size, nb_classes, X_train, X_test, X_test_holdout, score, flower_count_df, notes)

    model_summary_plots(history, save_output_root, model_type, name_time)

    # serialize model to JSON
    model_json = model_fitted.to_json()
    with open('{}{}_{}/model.json'.format(save_output_root, model_type, name_time), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    with h5py.File('{}{}_{}/{}_{}.h5'.format(save_output_root, model_type, name_time, model_type, name_time), 'w') as f: model_fitted.save('{}{}_{}/{}_{}.h5'.format(save_output_root, model_type, name_time, model_type, name_time))

    # Save model history to pickle
    f = open('{}{}_{}/history.pkl'.format(save_output_root, model_type, name_time), 'wb')
    for obj in [history]:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    # Save df of species indices, image counts per species
    flower_count_df.to_pickle('{}{}_{}/flower_count_df.pkl'.format(save_output_root, model_type, name_time))

    # Save predicted probabilities
    np.save('{}{}_{}/predicted_probas'.format(save_output_root, model_type, name_time), ypred, allow_pickle=True)

    # write_folder_to_bucket(new_root)



if __name__ == '__main__':
    start_time = datetime.datetime.now()
    name_time = time.time()

    # Model descriptors, parameters
    save_output_root = '../model_outputs/'
    model_type = "ResNet50"
    input_shape = (224,224,3)
    epochs = 45
    batch_size = 26
    notes = "SGD; learning rate: .001. Changed steps per epoch from len(x_train)/ batch size to just len(x_train)"

    # Load data from saved numpy array
    data = np.load('flowers_224.npz')
    x, y = read_data(data)

    # Describe data, make pandas df for counts of images and numerical label for species categories
    flower_count_df, class_labels = class_counts_specifications(y)
    nb_classes = len(flower_count_df)

    # Train test validation split
    X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout = train_validation_split(x, y)
    print('{} classes of flowers'.format(len(flower_count_df)))

    # Convert numerical y to one-hot encoded y
    Y_train, Y_test, Y_test_holdout = convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes)

    # Build CNN model
    final_model, model_summary = build_cnn_resnet_50(input_shape)

    # Fit CNN model
    ypred, model_fitted, history = fit_model_resnet50(X_train, X_test, Y_train, Y_test, save_output_root, model_type, name_time, batch_size, epochs, input_shape)

    finish_time = datetime.datetime.now()

    # Get predicted probabilities and evaulate model fit when fitted model run on validation hold out data set
    # test_predictions, score = predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout)

    # ytest_report = sklearn_stats(y_test, ypred, class_labels)

    test_predictions, score = predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout)
    # y_holdout_report = sklearn_stats(y_test_holdout, test_predictions, class_labels)

    save_model(name_time, history, model_fitted, flower_count_df, save_output_root, start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, X_train, Y_train, X_test, Y_test, X_test_holdout, Y_test_holdout, score)


import numpy as np
import pandas as pd
import exifread
from os import listdir
from os.path import isfile, join
import re
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import folium
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
# from multi_column_encoder import MultiColumnLabelEncoder

# from https://gist.github.com/snakeye/fdc372dbf11370fe29eb
# based on https://gist.github.com/erans/983821

pd.options.display.max_rows = 400

def _get_if_exist(data, key):
    if key in data:
        return data[key]
    return None

def _convert_to_degrees(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / value.values[0].den
    m = float(value.values[1].num) / value.values[1].den
    s = float(value.values[2].num) / value.values[2].den

    return d + (m / 60.0) + (s / 3600.0)

def get_exif_location(exif_data):
    """
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None
    gps_latitude = _get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = _get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = _get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref = _get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degrees(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = _convert_to_degrees(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon
    # if gps_latitude_ref.values[0] != 'N':
    #     gps_latitude.values[0].num *= -1
    # if gps_longitude_ref.values[0] != 'E':
    #     gps_longitude.values[0].num *= -1
    return lat, lon
    # return gps_latitude, gps_longitude

def gps_to_array_map(img_root):
    '''
    Returns array containing: image name, latitude, longitude values for images contained within directory provided.
    '''
    resultlist = []
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    for filename in files:
        if not (filename.startswith('.')):
            if not (filename.startswith('None')):
                path = '{}{}'.format(img_root, filename)
                with open(path, 'rb') as f:
                    tags = exifread.process_file(f)
                    lat, lon = get_exif_location(tags)
                    img_cat = re.sub("\d+", "", filename).rstrip('.jpg')
                    # img_cat = img_cat[:-3]
                    img_cat = img_cat.rstrip("_")
                    img_cat = img_cat.replace("_", " ")
                    resultlist.append((filename, lat, lon, img_cat))
            location_arr = np.asarray(resultlist)
    # plot_img_locations(location_arr)
    return location_arr

def plot_img_locations(location_arr):
    flower_map = folium.Map(location = [39.74675277777778, -105.2436], zoom_start = 10, tiles="Stamen Terrain")
    # for gps label in zip(df['gps'], df['label']):
    #     for label in df['label'].unique():
    for i in range(len(location_arr)):
        lat = location_arr[i][1]
        lon = location_arr[i][2]
        category = location_arr[i][3]
        folium.CircleMarker(location = [lat, lon],radius = 5, popup = category,
                    fill_color='#ff5050', ).add_to(flower_map)
        # folium.Marker(location = [lat, lon], popup = category).add_to(flower_map)
    return flower_map.save("../maps/flower_map.html")

def make_plant_instances(location_arr):
    '''
    Groups individual images into plant instances (i.e., multiple images were taken of the same plant) based on GPS location.
    Input: numpy array containing filename, latitude, longitude, plant species label
    Output: pandas dataframe containing filename, lat (latitude), lon (longitude), gps (tuple of lat, lon), and gps_instances (numerically encoded plant instances based on matching gps locations)
    '''
    location_df = pd.DataFrame({'filename': location_arr[:,0], 'lat': location_arr[:,1], 'lon': location_arr[:,2], 'label': location_arr[:,3]})
    location_df['gps'] = list(zip(location_df.lat, location_df.lon))
    le = LabelEncoder()
    location_df['gps_instances'] = le.fit_transform(location_df['gps'])
    return location_df

def check_equal(lst):
    return lst.count(lst[0]) == len(lst)

def check_all_same_species(location_df):
    result_list = []
    one_longs = []
    all_same = True
    for i in range(location_df['gps_instances'].min(), location_df['gps_instances'].max()):
        subset = location_df[location_df['gps_instances'] == i]
        same_species = check_equal(list(subset['label']))
        if same_species == False:
            result_list.append('gps_instance {} has more than one species.'.format(i))
            all_same = False
        # else:
        #     subset_len.append((i, len(subset)))
    total_instances = location_df['gps_instances'].nunique()
    min_instances = location_df['gps_instances'].value_counts().min()
    max_instances = location_df['gps_instances'].value_counts().max()
    # if len(location_df[location_df['gps_instances']].value_counts()) == 1:
    #     one_longs.append(df['gps_instances'])
    # instances_with_one_img =  set(one_longs)
    # proportion_with_one_img = instances_with_one_img / total_instances
    print('GPS Instances (unique plants): {}\n Min images per plant: {}\n Max images per plant: {}'.format(total_instances, min_instances, max_instances))
    if all_same == True:
        print('All instances contain only one species. Hooray!')
    else:
        print(result_list)
    return all_same
    # results = find_rows(location_arr)
    # for pair in results:
    #     location_arr[location_arr['']]
    # for i in range(len(location_arr)):
    #     coords = ((location_arr[i][1], location_arr[i][2]))

def find_rows(location_arr):
    iterable = zip(location_arr[:,0], location_arr[:,1])
    result_list = []
    for thing in combinations(iterable, 2):
        print(thing[0][1], thing[1][1])
        if float(thing[0][1]) == float(thing[1][1]):
            result_list.append(thing)
    return result_list

    # a = location_arr
    # b = np.copy(location_arr)
    # dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    #
    # a_view = np.ascontiguousarray(a).view(dt).ravel()
    # b_view = np.ascontiguousarray(b).view(dt).ravel()
    #
    # sort_b = np.argsort(b_view)
    # where_in_b = np.searchsorted(b_view, a_view,
    #                              sorter=sort_b)
    # where_in_b = np.take(sort_b, where_in_b)
    # which_in_a = np.take(b_view, where_in_b) == a_view
    # where_in_b = where_in_b[which_in_a]
    # which_in_a = np.nonzero(which_in_a)[0]
    # return np.column_stack((which_in_a, where_in_b))

def unique_rows(location_arr):
    result_arr = find_rows(location_arr)
    final_list = []
    for i in range(len(result_arr)):
        if result_arr[i][0] != result_arr[i][1]:
            final_list.append(list(result_arr[i]))
            print(list(result_arr[i]))
    return np.array(final_list)
    # result_arr = np.array(result_arr)
    # return result_arr
    # result_arr = result_arr[result_arr[:,0] ]
    # for i in len(result_arr):
    #
    # return result_arr

# def get_exif_data(filename):
#     with open(filename, 'rb') as f:
#         tags = exifread.process_file(f)
#         lat, lon = get_exif_location(tags)
#     return lat, lon


if __name__ == '__main__':
    lat, lon = get_exif_location(exif_data)
    
    import boto3
import os
from os import listdir
from os.path import isfile, join

access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

# s3 = boto3.client('s3')
# response = s3.list_buckets()
# Get a list of all bucket names from the response
# buckets = [bucket['Name'] for bucket in response['Buckets']]

def retrieve_from_bucket(file):
    """
    Download matrices from S3 bucket
    """
    conn = boto.connect_s3(access_key, access_secret_key)
    bucket = conn.get_bucket('depression-detect')
    file_key = bucket.get_key(file)
    file_key.get_contents_to_filename(file)
    X = np.load(file)
    return X

def write_to_bucket(bucket_name, filepath, filename):
    s3 = boto3.client('s3')
    print('Uploading {}...'.format(filename))
    s3.upload_file(filepath, bucket_name, filename)
    print('  {} uploaded - yay!'.format(filename))

def write_folder_to_bucket(root, bucket_name, bucket_folder):
    s3 = boto3.client('s3')
    files = [f for f in listdir(root) if isfile(join(root, f))]
    for thing in files:
        print('Uploading {}...'.format(thing))
        write_to_bucket(bucket_name, '{}{}'.format(root, thing), '{}/{}'.format(bucket_folder, thing))
    print('  {} uploaded - HUZZAH!'.format(root))


# if __name__ == '__main__':
    # retrieve_from_bucket('imgs_jpgs.zip')
    # write_to_bucket('capstonedatajen', '../imgs_for_readme_rsz.zip', 'imgs_for_readme_rsz.zip')
    # write_to_bucket('capstonedatajen', '../model_plots/test.txt', 'test.txt')
    # write_folder_to_bucket('../misclass_imgs/')


'''
Image preprocessing for CNN.
Input: jpg files exported from Mac Photo app
Output: Saved numpy array of images in specified shape for use in CNN. Corrects for class imbalance via image generation.
Images are resized to specified shape, cropped to square, then image generation is used to create flipped/rotated images.
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import os, cv2, re, PIL
from os import listdir
from os.path import isfile, join
from PIL import Image
from skimage import io
from skimage.transform import resize

from img_resize import my_image_resize
from img_resize import my_image_rename

np.random.seed(1337)  # for reproducibility


def image_categories(img_root):
    ''' A dictionary that stores the image path name and flower species for each image
    Input: image path names (from root directory)
    Output: dictionary 'categories'
    '''
    flower_dict = {}
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    # for path, subdirs, files in os.walk(resized_root):
        # print(path, subdirs, files)
    for name in files:
        # name = name.replace(' ', '')
        # name = name.replace('-', '_')
        if not (name.startswith('.')):
        #     if name != 'cnn_capstone.py':
            img_path = '{}{}'.format(img_root, name)
            # img_path = os.path.join(path, name)
            img_cat = re.sub("\d+", "", name).rstrip('_.jpg')
            # img_cat = img_cat[:-3]
            flower_dict[img_path] = img_cat
    return flower_dict

def _center_image(img, new_size=[256, 256]):
    '''
    Helper function. Takes rectangular image resized to be max length on at least one side and centers it in a black square.
    Input: Image (usually rectangular - if square, this function is not needed).
    Output: Image, centered in square of given size with black empty space (if rectangular).
    '''
    row_buffer = (new_size[0] - img.shape[0]) // 2
    col_buffer = (new_size[1] - img.shape[1]) // 2
    centered = np.zeros(new_size + [img.shape[2]], dtype=np.uint8)
    centered[row_buffer:(row_buffer + img.shape[0]), col_buffer:(col_buffer + img.shape[1])] = img
    return centered

def resize_image_to_square(img, new_size=((256, 256))):
    '''
    Resizes images without changing aspect ratio. Centers image in square black box.
    Input: Image, desired new size (new_size = [height, width]))
    Output: Resized image, centered in black box with dimensions new_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*new_size[1]/img.shape[0]),new_size[1])
    else:
        tile_size = (new_size[1], int(img.shape[0]*new_size[1]/img.shape[1]))
    # print(cv2.resize(img, dsize=tile_size))
    return _center_image(cv2.resize(img, dsize=tile_size), new_size)

def crop_image(img, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def process_images(file_paths_list, resize_new_size=[256,256], crop_size=[224, 224]):
    '''
    Input: list of file paths (images)
    Output: numpy array of processed images: normalized, resized, centered)
    '''
    x = []

    for file_path in file_paths_list:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image_to_square(img, new_size=resize_new_size)
        img = crop_image(img, crop_size=crop_size)
        x.append(img)
    x = np.array(x)
    return x


if __name__ == '__main__':
    img_root = '../imgs_jpgs/'
    # Rename files exported from Mac Photo app (in place)
    my_image_rename(img_root)
    # Create y (labels) and file list (x) from image names
    y_dict = image_categories(img_root)
    y = list(y_dict.values())
    file_list = list(y_dict.keys())
    # with Pool(4) as p:
    #     p.map(process_images(file_list, resize_new_size=[256,256], crop_size=[224, 224]), file_list)
    image_array = process_images(file_list, resize_new_size=[256,256], crop_size=[224, 224])
    np.savez('flowers_224_2.npz', image_array, y)


from __future__ import print_function
import numpy as np
import os, cv2, re, PIL
from os import listdir
from os.path import isfile, join
import re
import PIL
from PIL import Image
from skimage import io
import scipy.misc

def my_image_rename(img_root):
    '''Renames files without spaces in the name"'''
    pathiter = (os.path.join(img_root, name) for root, subdirs, files in os.walk(img_root) for name in files)
    for path in pathiter:
        # newname = path.replace(" ", "")
        # newname = path.replace("-", "_")
        newname = path.replace("_200", "")
        # newname = path.replace(" ", "")
        # newname = path.replace("-", "_")
        # newname = path.replace("arnica_jpg", "sand_lily")
        if newname != path:
            os.rename(path, newname)

def my_image_resize(basewidth, img_root, target_root):
    '''
    Input: desired basewidth for resized images, path for folder containing original images, name for new path for images
    '''
    os.mkdir('{}'.format(target_root))
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
            # if name != 'cnn_capstone.py':
                img = Image.open('{}{}'.format(img_root, name))
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
                img.save('{}/{}.jpg'.format(target_root, name[:-4]))

def crop_image(img, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def square_thumbnails(img_root, target_root):
    # my_image_resize(200, img_root, target_root)
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
                img_full_path = '../capstone_web_app/static/images/img_dict/{}'.format(name)
                img = cv2.imread(img_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_image(img, (200, 200))
                scipy.misc.toimage(img).save('{}/{}'.format(target_root, name))

                # cv2.imwrite('{}/{}'.format(target_root, name), img)
                # img.save('{}/{}.jpg'.format(target_root, name[:-4]))
def resize_thumbnails(basewidth, img_root, target_root, crop_size):
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
            # if name != 'cnn_capstone.py':
                img_full_path = '../capstone_web_app/static/images/img_dict/{}'.format(name)
                img = cv2.imread(img_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = Image.open('{}{}'.format(img_root, name))
                wpercent = (basewidth / float(img.shape[0]))
                hsize = int((float(img.shape[1]) * float(wpercent)))
                img = cv2.resize(img, (200, 267))
                img = crop_image(img, crop_size)
                scipy.misc.toimage(img, cmin=0.0, cmax=...).save('{}/{}'.format(target_root, name))

if __name__ == '__main__':
    img_root = '../capstone_web_app/static/images/img_dict/'
    target_root = '../capstone_web_app/static/images/thumbs_200w'
    os.mkdir('{}'.format(target_root))
    # my_image_rename(img_root)
    resize_thumbnails(200, img_root, target_root, (200, 200))
    # my_image_resize(200, img_root, target_root)
    # square_thumbnails(200, img_root, target_root, (200, 200))

    

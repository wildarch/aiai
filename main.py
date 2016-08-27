from stolen_network import Network
from wand.image import Image
import os
import numpy as np

training_dir = '../GameBot/training_data/'
test_dir = '../GameBot/test_data/'
img_width = 6

def get_px_values(image):
    vals = []
    for rows in image:
        for item in rows:
            vals.append(item.red)
    #return np.array(vals).reshape(1, len(vals))
    return np.array(vals).reshape(36,1)

def color_to_repr(col):
    if col == 'b': r = [1, 0, 0]
    elif col == 'w': r = [0, 1, 0]
    elif col == 'x': r = [0, 0, 1]
    else: return ValueError
    return np.array(r).reshape(3,1)

def color_to_test_repr(col):
    if col == 'b': return 0
    elif col == 'w': return 1
    elif col == 'x': return 2
    return ValueError

def img_from_path(path):
    return Image(filename=path)

def get_data_from_images(images):
    for img in images:
        img.resize(img_width, img_width)
    return map(get_px_values, images)

training_files = list(map(lambda file: os.path.join(training_dir, file), os.listdir(training_dir)))
colors = map(lambda file: file[-5], training_files)
images = list(map(img_from_path, training_files))
training_data = list(zip(get_data_from_images(images), map(color_to_repr, colors)))

test_files = list(map(lambda file: os.path.join(test_dir, file), os.listdir(test_dir)))
colors = map(lambda file: file[-5], test_files)
images = list(map(img_from_path, test_files))
test_data = list(zip(get_data_from_images(images), map(color_to_test_repr, colors)))

network = Network([img_width*img_width, 12, 3])
network.SGD(training_data, 30, 10, 3)
for file, output, input in zip(test_files, network.evaluate(test_data), test_data):
    if(output == 0):
        print("File {0} failed detection".format(file))
        raw_out = network.feedforward(input[0])
        print("Output was {0}. Should be {1}".format(raw_out, input[1]))

import numpy as np
import scipy.misc
from os import listdir
from os.path import isfile, join
import imageio

import time

def make_generator(path, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        # files = range(n_files)
        files = listdir(path)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = imageio.imread(path + i)
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='../sdgan/data/shoes4k/'):
    return (
        make_generator(data_dir+'train/', batch_size),
        make_generator(data_dir+'valid/', batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print( "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        print()
        if i == 50:
            break
        t0 = time.time()

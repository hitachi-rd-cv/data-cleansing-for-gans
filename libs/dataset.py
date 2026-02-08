# Modified from FreezeD (https://github.com/sangwoomo/FreezeD/tree/master/stylegan).
#
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from io import BytesIO
import multiprocessing

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, labels, resolution=256):
        self.path = path
        self.resolution = resolution
        self.transform = transform
        self.labels = labels

        # Temporarily open LMDB to get the length
        self.length = self.get_length(path)

        assert len(self.labels) == self.length, 'Number of labels must match number of images: {} != {}'.format(len(self.labels), self.length)

    def __len__(self):
        return self.length

    @staticmethod
    def get_length(path):
        env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
        env.close()

        return length

    def open_lmdb(self):
        if not hasattr(self, 'env'):
            self.env = lmdb.open(
                self.path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False, buffers=True)

    def __getitem__(self, index):
        self.open_lmdb()  # Ensure LMDB is open
        key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return index, img

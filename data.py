"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import cv2
import random

def default_loader(path):
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    return cv2.imread(path)


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Image.MAX_IMAGE_PIXELS = None # for my hist images

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def getFiles(dirName:str) -> list:
    '''
    Function to get a list of files from a directory and all subdirectories

    ### Example:
    ```
    dirName = '/Users/krunal/desktop/code/pyt'
    listOfFiles = getFiles(dirName)
    ```
    '''
    listOfFile = os.listdir(dirName)
    completeFileList = list()
    for file in listOfFile:
        completePath = os.path.join(dirName, file)
        if os.path.isdir(completePath):
            completeFileList = completeFileList + getFiles(completePath)
        else:
            completeFileList.append(completePath)

    return completeFileList


class ImageFolder(data.Dataset):
    '''
    TODO: implement return_paths=True
    '''
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        self.transform = transform
        self.files = getFiles(root)
        
        if len(self.files) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        img_A_path = self.files[index % len(self.files)]
        img_A = default_loader(img_A_path)

        folder_B, file_A = os.path.split(img_A_path)
        files_B = getFiles(folder_B)
        files_B.remove(img_A_path)
        index_B = random.randint(0, len(files_B)-1)
        img_B = files_B[index_B]
        
        img_B = default_loader(img_B)
        
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

    def __len__(self):
        return len(self.files)

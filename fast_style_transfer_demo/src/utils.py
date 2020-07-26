"""
DOCSTRING
"""
import numpy
import os
import scipy.misc
import sys

def check_version():
    """
    DOCSTRING
    """
    if sys.version_info[0] != 2:
        err_str = (
            "This project only supports Python 2! Either run using "
            "Python 2 or submit a pull request to "
            "https://github.com/lengstrom/fast-style-transfer/ "
            "to make the project version neutral!")
        raise Exception(err_str)

def exists(p, msg):
    assert os.path.exists(p), msg

def get_img(src, img_size=False):
    """
    DOCSTRING
    """
    img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = numpy.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def list_files(in_path):
    """
    DOCSTRING
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

def save_img(out_path, img):
    """
    DOCSTRING
    """
    img = numpy.clip(img, 0, 255).astype(numpy.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    """
    DOCSTRING
    """
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

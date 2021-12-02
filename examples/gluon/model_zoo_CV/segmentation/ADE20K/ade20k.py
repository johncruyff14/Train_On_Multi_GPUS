"""Prepare ADE20K dataset"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import download, makedirs

TRAIN_DATA_DIR = os.getenv("_TRAIN_DATA_DIR")
if TRAIN_DATA_DIR is None:
    _DATA_SETS_DIR = '~/.mxnet/datasets'
    _TARGET_DIR = os.path.expanduser(_DATA_SETS_DIR + '/ade')
else:
    _DATA_SETS_DIR = None
    _TARGET_DIR = TRAIN_DATA_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python setup_ade20k.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('ADEChallengeData2016', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('release_test', 'e05747892219d10e9243933371a497e905a4860c'),]

    # Check the existence of Dataset:
    flag = True
    for data,_ in _AUG_DOWNLOAD_URLS:
        if not os.path.isdir(path + '/' + data):
            flag = False

    if flag:
        return

    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for data, checksum in _AUG_DOWNLOAD_URLS:
        url = 'http://data.csail.mit.edu/places/ADEchallenge/' + data + '.zip'
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    if _DATA_SETS_DIR:
        makedirs(os.path.expanduser(_DATA_SETS_DIR))

    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    download_ade(_TARGET_DIR, overwrite=False)

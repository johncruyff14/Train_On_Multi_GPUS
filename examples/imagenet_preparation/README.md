# ImageNet dataset preparation

This example shows how to download and prepare ImageNet (or other custom) dataset to use with MXNet.

## Contents

1. [Download ImageNet dataset](#download-imagenet)
2. [Create lst files](#create-lst-files)
3. [Create RecordIO files](#create-recordio-files)
4. [Next steps](#next-steps)

## Download ImageNet dataset

 - Download the images from http://image-net.org/download-images.
 - Extract the training and valiation data. Here we assume that training images location is
   `/data/imagenet/train-jpeg` and validation images location is `/data/imagenet/val-jpeg`

```bash
mkdir -p /data/imagenet/train-jpeg
mv ILSVRC2012_img_train.tar /data/imagenet/train-jpeg/
cd /data/imagenet/train-jpeg
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd -
mkdir /data/imagenet/val-jpeg
mv ILSVRC2012_img_val.tar /data/imagenet/val-jpeg/
cd /data/imagenet/val-jpeg
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Create lst files

To make a RecordIO file that will be used in training, we first need to create `.lst` files for both training and validation images. To do that, we can use MXNet's `im2rec.py` utility:
```bash
python /opt/mxnet/tools/im2rec.py --list --recursive train /data/imagenet/train-jpeg
python /opt/mxnet/tools/im2rec.py --list --recursive val /data/imagenet/val-jpeg
```

## Create RecordIO files

RecordIO is a input file format used by MXNet to achieve high performance when loading data (see [http://mxnet.io/architecture/note_data_loading.html] for further details).

We will make 2 RecordIO files - with training and validation data, respectively. To achieve that, we will again use MXNet's `im2rec.py` utility:

```bash
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 train /data/imagenet/train-jpeg
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 val /data/imagenet/val-jpeg
```

Here we packed the uncompressed JPEGs into RecordIO files  and used 40 threads to accelerate conversion.
As a result, we got 4 files `train.rec`, `train.idx`, `val.rec` and `val.idx`.

# Next steps

- Image classification example


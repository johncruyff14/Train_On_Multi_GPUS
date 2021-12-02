#!/bin/sh

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=andreii&password=AndreiI@NVidia&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=$1

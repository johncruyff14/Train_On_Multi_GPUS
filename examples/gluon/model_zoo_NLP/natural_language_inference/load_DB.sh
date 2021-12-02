#!/usr/bin/env bash

DATADIR=$1
export SNLI=snli_1.0

pip install nltk

if [ -d "${DATADIR}" ]; then
  if [ -d "${DATADIR}/${SNLI}" ]; then
    exit 1
  fi
else
  mkdir ${DATADIR}
fi

SNLI_ZIP=${DATADIR}/${SNLI}.zip
curl https://nlp.stanford.edu/projects/snli/${SNLI}.zip -o ${SNLI_ZIP}

# For now we do not extract __MACOSX
unzip ${SNLI_ZIP} ${SNLI}/* -d ${DATADIR}
rm ${SNLI_ZIP}

for split in train dev test; do 
  python ../preprocess.py --input ${DATADIR}/${SNLI}/${SNLI}_$split.txt --output ${DATADIR}/${SNLI}/$split.txt
done




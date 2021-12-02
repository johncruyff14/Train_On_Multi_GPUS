#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# This script produces the results from Table 1 in the following paper:
# Bag of Tricks for Efficient Text Classification, arXiv 1607.01759, 2016

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

DATASET=(
  ag_news
  sogou_news
  dbpedia
  yelp_review_polarity
  yelp_review_full
  yahoo_answers
  amazon_review_full
  amazon_review_polarity
)

ID=(
  0Bz8a_Dbh9QhbUDNpeUdjb0wxRms # ag_news
  0Bz8a_Dbh9QhbUkVqNEszd0pHaFE # sogou_news
  0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k # dbpedia
  0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg # yelp_review_polarity
  0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0 # yelp_review_full
  0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU # yahoo_answers
  0Bz8a_Dbh9QhbZVhsUnRWRDhETzA # amazon_review_full
  0Bz8a_Dbh9QhbaW12WVVZS2drcnM # amazon_review_polarity
)

# These learning rates were chosen by validation on a subset of the training set.
LR=( 0.25 0.5 0.5 0.1 0.1 0.1 0.05 0.05 )

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

# Small datasets first
echo "Searching for $1"
for i in {0..7}
do
  if [ ! ${DATASET[i]} = $1 ]; then
    continue
  fi

  if [ ! -f "${DATADIR}/$1.train" ]; then
    echo "Downloading dataset $1"
    if [ $i -gt 0 ]; then
      curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${ID[i]}" > /tmp/intermezzo.html
      curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > "${DATADIR}/$1_csv.tar.gz"
    else
      wget -c "https://drive.google.com/uc?export=download&id=${ID[i]}" -O "${DATADIR}/$1_csv.tar.gz"
    fi

    tar -xzvf "${DATADIR}/$1_csv.tar.gz" -C "${DATADIR}" --no-same-owner
    rm "${DATADIR}/$1_csv.tar.gz"
    cat "${DATADIR}/$1_csv/train.csv" | normalize_text > "${DATADIR}/$1.train"
    rm "${DATADIR}/$1_csv/train.csv"
    cat "${DATADIR}/$1_csv/test.csv" | normalize_text > "${DATADIR}/$1.test"
    rm "${DATADIR}/$1_csv/test.csv"
  fi
  break
done



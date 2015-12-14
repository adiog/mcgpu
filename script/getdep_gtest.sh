#!/bin/bash
# Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
#   created:  Sat 12 Dec 2015 04:52:48 PM CET
#   modified: Sun 13 Dec 2015 10:58:33 PM CET

# BASH_CLEANUP {{{
BASH_CLEANUP_FILE=`mktemp`
trap BASH_CLEANUP EXIT

function BASH_CLEANUP() {
  BASH_CLEANUP_FILE_REV=`mktemp`
  tac $BASH_CLEANUP_FILE > $BASH_CLEANUP_FILE_REV
  . $BASH_CLEANUP_FILE_REV
  rm $BASH_CLEANUP_FILE $BASH_CLEANUP_FILE_REV
}

function BASH_FINALLY() {
  echo "$*" >> $BASH_CLEANUP_FILE
}

function BASH_MKTEMP() {
  BASH_TMP=`mktemp`
  echo "rm $BASH_TMP" >> $BASH_CLEANUP_FILE
  echo $BASH_TMP
}

function BASH_MKTEMP_DIR() {
  BASH_TMP=`mktemp -d`
  echo "rm -fr $BASH_TMP" >> $BASH_CLEANUP_FILE
  echo $BASH_TMP
}
# }}}

TMP=`BASH_MKTEMP_DIR`
cd $TMP
git clone https://github.com/google/googletest.git
cd googletest/googletest
mkdir build
cd build
cmake ..
make
sudo cp -v *.a /usr/lib
sudo cp -vr ../include/gtest /usr/include
cd -


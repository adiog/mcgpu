#!/bin/bash
# Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
#   created:  Sat 12 Dec 2015 04:52:48 PM CET
#   modified: Sat 12 Dec 2015 04:58:00 PM CET

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
git clone https://github.com/google/benchmark.git
cd benchmark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
make
sudo make install
cd -


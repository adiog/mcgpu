#!/bin/bash
# Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
#   created:  Fri 11 Dec 2015 05:49:50 AM CET
#   modified: Fri 11 Dec 2015 06:01:31 PM CET

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

ABS_PATH_CMAKE=$1
ABS_PATH_TARGET=$2

BUILD_TMP=`BASH_MKTEMP_DIR`
cd ${BUILD_TMP}
cmake ${ABS_PATH_CMAKE}
make
./gen_device_hpp > ${ABS_PATH_TARGET}


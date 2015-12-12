#!/bin/bash
# Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
#   created:  Thu 10 Dec 2015 11:34:31 PM CET
#   modified: Fri 11 Dec 2015 06:01:01 PM CET

RESULT=$1
shift 1
rm -f ${RESULT}

for file in $*; do
    cat $file >> ${RESULT}
done


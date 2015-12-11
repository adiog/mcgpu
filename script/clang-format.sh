#!/bin/bash
# Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
#   created:  Fri 11 Dec 2015 05:55:24 PM CET
#   modified: Fri 11 Dec 2015 07:38:20 PM CET

clang-format -style="{BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 80, ConstructorInitializerAllOnOneLineOrOnePerLine: true, AccessModifierOffset: -2 }" $*

for file in $*; do
  if [[ -e $file ]]; then
    sed -e "s#\s*<\s*<\s*<\(.*\)>\s*>\s*>\s*#<<<\1>>>#" -i $file
  fi
done


#!/bin/bash
set -eu
rm -f -r build
mkdir build
cd build
#~ cmake ../ -DCMAKE_BUILD_TYPE=DEBUG
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE

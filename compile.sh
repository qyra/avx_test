#!/bin/bash
set -eu
cd build
make
cp avx_test ../

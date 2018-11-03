#!/bin/bash
cd build_debug
cmake ..
cmake --build ./ -- -j4
cd ..


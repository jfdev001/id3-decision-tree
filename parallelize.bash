#!/bin/bash

cat | xargs -L 1 -I CMD -P 64 bash -c CMD


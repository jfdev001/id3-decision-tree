#!/bin/bash

cat | xargs -L 1 -I CMD -P 16 bash -c CMD


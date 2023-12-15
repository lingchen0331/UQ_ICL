#!/bin/bash

python sources/cola.py --model 70b --num_demos_per_class 2 --num_demos 4 --sampling_strategy "random" --iter_demos 4 --load8bits "False"
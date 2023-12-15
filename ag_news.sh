#!/bin/bash

python sources/ag_news.py --model 7b --num_demos_per_class 1 --num_demos 4 --sampling_strategy "class" --iter_demos 4 --load8bits "False"
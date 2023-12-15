#!/bin/bash

python sources/financial_phrasebank.py --model 70b --num_demos_per_class 1 --num_demos 3 --sampling_strategy "random" --iter_demos 4 --load8bits "False"
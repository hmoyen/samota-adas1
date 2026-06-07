#! /bin/bash

REPEATS=$1
python3.11 PFES_falsification.py --size 30 --niterations 30 --nruns $REPEATS --optalg RANDOM  --logdir out
python3.11 PFES_falsification.py --size 30 --niterations 30 --nruns $REPEATS --optalg NSGA3  --logdir out
python3.11 PFRL_falsification.py --nepisodes 900 --nruns $REPEATS --logdir out
python3.11 FOC_falsification.py --size 30 --totbudget 900 --nruns $REPEATS --logdir out

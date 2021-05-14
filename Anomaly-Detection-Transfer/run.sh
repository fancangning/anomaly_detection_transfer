#!/bin/sh
#****************************************************************#
# ScriptName: run.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-03-23 14:26
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-03-23 14:26
# Function: 
#***************************************************************#

python Test.py --gpu 0 --AdversarialLossWeight 5e-4 &
python Test.py --gpu 1 --AdversarialLossWeight 1e-4 &
python Test.py --gpu 2 --AdversarialLossWeight 5e-3 &
python Test.py --gpu 3 --AdversarialLossWeight 1e-3 &
python Test.py --gpu 4 --AdversarialLossWeight 5e-2 &
python Test.py --gpu 5 --AdversarialLossWeight 1e-2 &
python Test.py --gpu 6 --AdversarialLossWeight 0.5 &
python Test.py --gpu 7 --AdversarialLossWeight 0.1 &
#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..


chmod -R 700 ${basename}/../tasks/task_viper
# If you do not specify the GPU ID, it will use all the GPU.
python ${basename}/../tasks/task_viper/train_test.py --gpu=0
sleep 20

chmod -R 700 ${basename}/../tasks/taskname
python ${basename}/../tasks/taskname/train_test.py --gpu=0
sleep 20

#!/bin/bash

basename=$(dirname $(readlink -f $0))

chmod -R 700 ${basename}/..


chmod -R 700 ${basename}/../tasks/task_video
# If you do not specify the GPU ID, it will use all the GPU.

# You need four GPUs to run this job.
python ${basename}/../tasks/task_video/train_test.py --gpu=0,1,2,3
sleep 20

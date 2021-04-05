#!/bin/bash
FAILURE=false

python training/run_experiment.py --data_class=FakeImageData --model_class=CNN --conv_dim=32 --fc_dim=16 --loss=cross_entropy --num_workers=4 --max_epochs=4 || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Test for run_experiment.py passed"
exit 0

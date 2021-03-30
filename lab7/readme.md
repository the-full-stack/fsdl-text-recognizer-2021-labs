# Lab 7: Paragraph Recognition

*Thanks to [Saurabh Bipin Chandra](https://www.linkedin.com/in/srbhchandra/) for extensive work on this lab!*

In this lab, we will do several things:
- Move from training on synthetic line data to training on real data -- the `IAMLines` data module
- Move from training on line data to training on paragraph data -- the `IAMParagraphs` data module
- Automatically save the final model
- Introduce `ParagraphTextRecognizer` class to load the model and run inference that we can use in production

## Instructions

1. We are now using newer versions of `pytorch-lightning` and `wandb` packages, so make sure to run `make pip-tools` to update your setup.
2. We are also using `git-lfs` to store model weights now, so install it with `git lfs install`.

## Project Structure

```
├── text_recognizer
│   ├── artifacts
│   │   └── paragraph_text_recognizer
│   │       ├── config.json                           <-- NEW FILE
│   │       ├── model.pt                              <-- NEW FILE
│   │       └── run_command.txt                       <-- NEW FILE
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── emnist_essentials.json
│   │   ├── emnist_lines2.py
│   │   ├── emnist_lines.py
│   │   ├── emnist.py
│   │   ├── iam_lines.py
│   │   ├── iam_original_and_synthetic_paragraphs.py  <-- NEW FILE
│   │   ├── iam_paragraphs.py                         <-- NEW FILE
│   │   ├── iam.py
│   │   ├── iam_synthetic_paragraphs.py               <-- NEW FILE
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   ├── sentence_generator.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   ├── ctc.py
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── transformer.py
│   │   └── util.py
│   ├── models
│   │   ├── cnn.py
│   │   ├── __init__.py
│   │   ├── line_cnn_lstm.py
│   │   ├── line_cnn.py
│   │   ├── line_cnn_simple.py
│   │   ├── line_cnn_transformer.py
│   │   ├── mlp.py
│   │   ├── resnet_transformer.py                    <-- NEW FILE
│   │   └── transformer_util.py
│   ├── paragraph_text_recognizer.py                 <-- NEW FILE
│   └── util.py
└── training
    ├── __init__.py
    ├── run_experiment.py
    ├── save_best_model.py                           <-- NEW FILE
    └── sweeps
        └── emnist_lines2_line_cnn_transformer.yml
```

## IAMLines

Training command:

```sh
python training/run_experiment.py --wandb --gpus=-1 --data_class=IAMLines --model_class=LineCNNTransformer --loss=transformer --num_workers=12 --accelerator=ddp --lr=0.001
```

Best run:

- Link: https://wandb.ai/srbhchandra/fsdl-text-recognizer-2021-labs/runs/2snh6jvw
- Results:
   - test_cer: 0.1176
   - val_cer: 0.09418
   - val_loss: 0.2252
   - num_epochs: 589

## IAMParagraphs

Training command:

```sh
python training/run_experiment.py --wandb --gpus=-1 --data_class=IAMOriginalAndSyntheticParagraphs --model_class=ResnetTransformer --loss=transformer --batch_size=16 --check_val_every_n_epoch=10 --terminate_on_nan=1 --num_workers=24 --accelerator=ddp --lr=0.0001 --accumulate_grad_batches=4
```

Best run:

- Link: https://wandb.ai/srbhchandra/fsdl-text-recognizer-2021-labs/runs/2x9f8e1b
- Results:
   - test_cer: 0.1754
   - val_cer: 0.07208
   - val_loss: 0.06945
   - num_epochs: 999

## Saving the final model

TODO

## ParagraphTextRecognizer

Now that we have our final model parameters and weights, we can load it in a class that we can use for inference in production.

The file for this is `text_recognizer/paragraph_text_recognizer.py`

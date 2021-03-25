# Lab 3: Using CNN + LSTM with CTC loss for line text recognition

## Goals

- Introduce `LineCNNSimple`, a model that can read multiple characters in an image
- Make this model more efficient as `LineCNN`
- Introduce CTC loss with `LitModelCTC`
- Introduce an LSTM layer on top of CNN with `LineCNNLSTM`

## Follow along

```
git pull
cd lab3
```

## New files

```
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── emnist_essentials.json
│   │   ├── emnist_lines.py
│   │   ├── emnist.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   ├── sentence_generator.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   ├── ctc.py              <-- NEW
│   │   ├── __init__.py
│   │   ├── metrics.py          <-- NEW
│   │   └── util.py             <-- NEW
│   ├── models
│   │   ├── cnn.py
│   │   ├── __init__.py
│   │   ├── line_cnn_lstm.py    <-- NEW
│   │   ├── line_cnn.py         <-- NEW
│   │   ├── line_cnn_simple.py  <-- NEW
│   │   └── mlp.py
│   └── util.py
```

## LineCNNSimple: Reading multiple characters at once

Now that we have a dataset of lines and not just single characters, we can apply our convolutional net to it.

Let's look again at `notebooks/02-look-at-emnist-lines.ipynb` for a reminder of what the data looks like.

The first model we will try is a simple wrapper around `CNN` that applies it to each square slice of the input image in sequence: `LineCNNSimple`.

Go ahead and take a look at the code.

We can train this with

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=28
```

With this, we can get to 90% accuracy.

### Loss Function

Note that we are still using the `BaseLitModel` with the default `cross_entropy` loss function.
From reading [PyTorch docs](https://pytorch.org/docs/stable/nn.functional.html#cross-entropy) on the function, we can see that it accepts multiple labels per example just fine -- it's called "K-dimensional" loss.

### Changing window_stride

Let's go ahead and change window_stride, such that we are sampling overlapping windows.

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=20
```

Oops! That errored. We need add one more flag: `--limit_output_length`, since with the new stride, our model outputs a different length sequence than our ground truth expects (this will not be a problem once we start using CTC loss).

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

This won't get to as high of an accuracy (I max out at <60%), because the dataset does not actually have overlapping characters, whereas our model expects them.

### Changing overlap

To match our new `window_stride`, we can have our synthetic dataset overlap by 0.25:

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

This will get accuracy into the 80%'s.

### Variable-length overlap

We can see that if our model `window_stride` matches the character overlap in our data, it can train successfully.

Real handwriting has a variety of styles: some people write with characters close together, some far apart, and the width of different characters is also different.
To make our synthetic data more like this, we can set `--min_overlap=0 --max_overlap=0.33`.

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

As you probably expect, our model is not able to handle this non-uniform overlap amount.
Best accuracy I get is ~60%.

## LineCNN: making things more efficient

The simple implementation of a line-reading CNN above works fine, but it's highly inefficient if `window_stride` is less than `window_width`, because it send each window through the CNN separately.

We can improve on this with a fully-convolutional model, `LineCNN`.

Go ahead and take a look at the model code.

We can train a model on a fixed-overlap dataset:

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --limit_output_length
```

This performs just about the same as the previous model.

## CTC Loss

And now we get to the solution to our problem: CTC loss.

To use it, we introduce `CTCLitModel`, which is enabled by setting `--loss=ctc`.

Let's take a look at the code, and note a few things:

- Start, Blank, Padding tokens
- `torch.nn.CTCLoss` function
- `CharacterErrorRate`
- `.greedy_decode()`

Let's add CTC loss to our current model:

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --loss=ctc
```

This gets the CER down to ~18% in 10 epochs.

Best of all, we can now handle variable-overlap data:

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNN --window_width=28 --window_stride=18 --loss=ctc
```

This gets ~15% CER.

## Add LSTM

Lastly, we can add an LSTM on top of our `LineCNN` and see even more improvement.

The model is `LineCNNLSTM`, take some time to look at it.

We can train with it by running:

```sh
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNLSTM --window_width=28 --window_stride=18 --loss=ctc
```

## Homework

Two parts:

### Experiments

Play around with the hyperparameters of the CNN (`window_width`, `window_stride`, `conv_dim`, `fc_dim`) and/or the LSTM (`lstm_dim`, `lstm_layers`).

Better yet, edit `LineCNN` to use residual connections and other CNN tricks, or just change its architecture in some ways, like you did for Lab 2.

Feel free to edit `LineCNNLSTM` as well, get crazy with LSTM stuff!

### CTCLitModel

In your own words, explain how the `CharacterErrorRate` metric and the `greedy_decode` method work.

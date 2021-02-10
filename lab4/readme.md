# Lab 4: Recognize synthetic sequences with Transformers

Our goals are to introduce `LineCNNTransformer` and `TransformerLitModel`.

## LineCNNTransformer

In Lab 3, we trained a `LineCNN` + LSTM model with CTC loss.

In this lab, we will use the same `LineCNN` architecture as an "encoder" of the image, and then send it through Transformer decoder layers.

The [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
 ) for Transformer are not very good, and you might find our [simple Colab notebook](https://colab.research.google.com/drive/1swXWW5sOLW8zSZBaQBYcGQkQ_Bje_bmI) helpful.

 In the `LineCNNTransformer` class, pay attention to the `predict` method.

## TransformerLitModel

Nothing too fancy here.
We're now back to using cross entropy loss, but we're keeping the character error rate metrics from `CTCLitModel`.

## Training

I find that more epochs are necessary with the Transformer than with our LSTM+CTC model.
~30 epochs gives the same performance as we were able to obtain before.

I also changed the window width to 20 from 28, and window stride to 12, just because.

```
python training/run_experiment.py --max_epochs=40 --gpus=1 --num_workers=16 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNTransformer --window_width=20 --window_stride=12 --loss=transformer

DATALOADER:0 TEST RESULTS
{'test_acc': tensor(0.9022, device='cuda:0'),
 'test_cer': tensor(0.1749, device='cuda:0')}
```

## Homework

Standard stuff: try training with some different hyperparameters, explain what you tried.

There is also an opportunity to speed up the `predict` method that you could try.

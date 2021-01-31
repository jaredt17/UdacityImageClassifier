# Command Line Results
## Jared Teller's Image Classifier

# train.py
```
root@6cbb8e21aee8:/home/workspace/ImageClassifier# python train.py --gpu --save_dir /
Model running on:  cuda:0
Model Architecture:  vgg11
Input Features: 25088
Hidden units:  4096
Learning Rate:  0.001
Epochs:  5
TRAINING STARTED...

Epoch 1/5.. Train loss: 6.048.. Test loss: 2.413.. Test accuracy: 0.423
Epoch 1/5.. Train loss: 2.633.. Test loss: 1.411.. Test accuracy: 0.642
Epoch 1/5.. Train loss: 1.915.. Test loss: 0.934.. Test accuracy: 0.746
Epoch 1/5.. Train loss: 1.664.. Test loss: 0.760.. Test accuracy: 0.783
Epoch 2/5.. Train loss: 1.215.. Test loss: 0.653.. Test accuracy: 0.825
Epoch 2/5.. Train loss: 1.403.. Test loss: 0.610.. Test accuracy: 0.831
Epoch 2/5.. Train loss: 1.357.. Test loss: 0.616.. Test accuracy: 0.840
Epoch 2/5.. Train loss: 1.314.. Test loss: 0.533.. Test accuracy: 0.853
Epoch 3/5.. Train loss: 0.867.. Test loss: 0.534.. Test accuracy: 0.860
Epoch 3/5.. Train loss: 1.170.. Test loss: 0.454.. Test accuracy: 0.873
Epoch 3/5.. Train loss: 1.053.. Test loss: 0.439.. Test accuracy: 0.886
Epoch 3/5.. Train loss: 1.285.. Test loss: 0.412.. Test accuracy: 0.890
Epoch 4/5.. Train loss: 0.694.. Test loss: 0.454.. Test accuracy: 0.864
Epoch 4/5.. Train loss: 1.059.. Test loss: 0.467.. Test accuracy: 0.870
Epoch 4/5.. Train loss: 1.074.. Test loss: 0.428.. Test accuracy: 0.878
Epoch 4/5.. Train loss: 1.181.. Test loss: 0.462.. Test accuracy: 0.871
Epoch 5/5.. Train loss: 0.529.. Test loss: 0.428.. Test accuracy: 0.890
Epoch 5/5.. Train loss: 1.100.. Test loss: 0.393.. Test accuracy: 0.887
Epoch 5/5.. Train loss: 1.113.. Test loss: 0.433.. Test accuracy: 0.886
Epoch 5/5.. Train loss: 1.073.. Test loss: 0.421.. Test accuracy: 0.879

TRAINING COMPLETE.
Checkpoint Created at path:  /
```

# predict.py
```
root@6cbb8e21aee8:/home/workspace/ImageClassifier# python predict.py ./flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k 5 --category_names ./cat_to_name.json
Model running on:  cuda:0
Probabilities:  [ 0.77398872  0.20680928  0.00781976  0.0072024   0.00182629]
Classes:  ['1', '86', '83', '89', '51']
Class:  pink primrose ...  Probability:  77.3988723755 %
Class:  tree mallow ...  Probability:  20.6809282303 %
Class:  hibiscus ...  Probability:  0.781976059079 %
Class:  watercress ...  Probability:  0.720240268856 %
Class:  petunia ...  Probability:  0.182628678158 %
```



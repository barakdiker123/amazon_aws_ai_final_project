# How to create the model 

This model achieves 0.885 probability for accuracy

Epoch 4/4.. Train loss: 0.937.. Test loss: 0.404.. Test accuracy: 0.885

```bash 
 python3 train.py flowers --epochs 4 --gpu --hidden_units 512 --save_dir densenet201_model_v2 --learning_rate 0.005
```

for testing and predicting a picture 
```bash
python3 predict.py flowers/test/1/image_06743.jpg densenet201_model_v2/barak_model.pth --gpu
```

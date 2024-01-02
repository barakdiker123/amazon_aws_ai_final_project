

# Amazon Final project 
Here is the final project of AWS ML & AI 
This project was written By Barak-Nadav Diker 

## How to run this program 

For opening and running the jupyter notebook , simply do 

``` sh
cd PARENT DIR 
jupyter-notebook 
```

For Running the entire program one should do run the following 
### For training  

For using Efficientnet do this 

``` sh
cd PAR_DIR/src
python3 train.py flowers --gpu --epochs 7 --learning_rate 0.0006 --hidden_units 512 --save_dir efficientnet_v1 --arch efficientnet

```

`

For using alexnet do this 
``` sh

python3 train.py flowers --gpu --epochs 7 --learning_rate 0.00006 --hidden_units 512 --save_dir alexnet_v5 --arch alexnet 
```

`

For using densenejt do this 

``` sh

 python3 train.py flowers --gpu --epochs 6 --learning_rate 0.003 --save_dir dense_net_v3 
```

` 
 
### For predicting 

Here is a simple explanation and example on how to run the selected model 

``` sh

python3 predict.py flowers/test/1/image_06743.jpg densenet201_model_v2/barak_model.pth --gpu   
```

`
  

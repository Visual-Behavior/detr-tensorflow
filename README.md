# DETR : End-to-End Object Detection with Transformers (Tensorflow)

Tensorflow implementation of DETR : Object Detection with Transformers including code for inference, training, and finetuning. DETR is a promising model that brings widely adopted transformers to vision models. We believe that models based on convolution and transformers will soon become the default choice for most practitioners because of the simplicity of the training procedure: NMS and anchors free! Therefore this repository is a step toward making this type of architecture widely avaiable. 


<b>About this implementation:</b> https://arxiv.org/pdf/2005.12872.pdf <br>
<b>Torch implementation: https://github.com/facebookresearch/detr</b>

<img src="images/detr-figure.png"></img>

<b>About this implementation:</b> This repository includes codes to run an inference with the original model's weights (based on the PyTorch weights), to train the model from scratch (multi-GPU training support coming soon) as well as examples to finetune the model on your own dataset. Currently, unlike the PyTorch implementation, the training use fixed image sizes and a standard Adam optimizer with gradient norm.

Additionally, our logging system is based on https://www.wandb.com/ so you can get a nice visualization of your model performance!


## Current and upcoming features

- DETR Model ✔️
- Training ✔️
- Gradient aggregate ✔️
- Finetuning ✔️
- Evaluation ✔️
- Inference ✔️
- Jupyter notebook for finetuning ⌛
- Jupyter notebbok guide to setup your dataset ⌛
- Multi-GPU Training ⌛
- Training with weight decay ⌛
- Transformer attention head logging into wandb
- DETR-DC5
- DETR-R101
- DETR-DC5-R101


## Install

The code is currently tested with tensorflow 2.3.0 and python 3.7. The following dependencies are required

```
wandb
matplotlib
numpy
pycocotools
scikit-image
imageio
```

```
pip install -r requirements.txt
```


## Evaluation :

Run the following to evaluate the model using the pretrained weights:


```
python eval.py --datadir /path/to/coco
```

Outputs:
```
       |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
   box | 36.53 | 55.38 | 53.13 | 50.46 | 47.11 | 43.07 | 38.11 | 32.10 | 25.01 | 16.20 |  4.77 |
  mask |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

```

The result is not the same as reported in the paper because the evaluation is run on the <b>original image size</b> and not on the larger images. The original implementation resizes the images so that the shorter side is at least 800pixels and the longer side at most 1333.


## Finetune on your dataset

To finetune the model on a new dataset, we must get ride of the last layers that predicts bbox class and positions.

```python
# Input
image_input = tf.keras.Input((None, None, 3))

# Load the pretrained model
detr = get_detr_model(config, include_top=False, weights="detr", num_decoder_layers=6, num_encoder_layers=6)

# Setup the new layers
cls_layer = tf.keras.layers.Dense(len(CLASS_NAME), name="cls_layer")
pos_layer = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(4, activation="sigmoid"),
], name="pos_layer")

transformer_output = detr(image_input)
cls_preds = cls_layer(transformer_output)
pos_preds = pos_layer(transformer_output)

# Define the main outputs along with the auxialiary loss
outputs = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
outputs["aux"] = [ {"pred_logits": cls_preds[i], "pred_boxes": pos_preds[i]} for i in range(0, 5)]

detr = tf.keras.Model(image_input, outputs, name="detr_finetuning")
detr.summary()
return detr
```

The following script gives an example to finetune the model on a new dataset (VOC) with a real ```batch_size``` of 8 and a virtual ```target_batch``` size (gradient aggregate) of 32. ```--log``` is used to log the training into wandb. 


```
python finetune_voc.py --datadir /home/thibault/data/VOCdevkit/VOC2012 --batch_size 8 --target_batch 32  --log
```

## Training on COCO

(Multi GPU training comming soon)

```
python train_coco.py --datadir /home/thibault/data/VOCdevkit/VOC2012 --batch_size 8  --target_batch 32 --log
```


## Inference

Here is an exmaple to run inference with the webcam on your webcam

```
python webcam_inference.py 
```

## Acknowledgement

The pretrained weights of this models are originaly provide from the Facebook repository https://github.com/facebookresearch/detr and made avaiable in tensorflow in this repository: https://github.com/Leonardo-Blanger/detr_tensorflow

# FYP2020 Fine-tuning the model "Instance-aware Image Colorization"
This code is based mostly on the project:
https://github.com/ericsujw/InstColorization

This project aims to fine-tune one of the current state of the art models for image colourisation. The colourisation model in investigation utilises a 3 model system, a full mode colouristaion, which is the basis for the instance and fusion colourisation models. These models are fine-tuned on a comic book dataset to observe if this model can be customised for personal/other uses.

## Prerequisites
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
* Python3
* Pytorch >= 1.5
* Detectron2
* OpenCV-Python
* Pillow/scikit-image
* Please refer to the [env.yml](env.yml) for detail dependencies.

## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/ericsujw/InstColorization
cd InstColorization
```
2. Install [conda](https://www.anaconda.com/).
3. Install all the dependencies
```sh
conda env create --file env.yml
```
4. Switch to the conda environment
```sh
conda activate instacolorization
```
5. Install other dependencies
```sh
sh scripts/install.sh
```

## Dataset Preparation
### COCOStuff
1. Download and unzip the COCOStuff training set:
```sh
sh scripts/prepare_cocostuff.sh
```
2. Now the COCOStuff train set would place in [train_data](train_data).

### Your own Dataset
1. If you want to train on your dataset, you should change the dataset path in [scripts/prepare_train_box.sh's L1](scripts/prepare_train_box.sh#L1) and in [scripts/train.sh's L1](scripts/train.sh#L1).

## Pretrained Model
1. Download it from [google drive](https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh).
```sh
sh scripts/download_model.sh
```
2. Now the pretrained models would place in [checkpoints](checkpoints).

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `${DATASET_DIR}` folder.
```sh
sh scripts/prepare_train_box.sh
```
All the prediction results would save in `${DATASET_DIR}_bbox` folder.

## Training the Instance-aware Image Colorization model
Simply run the following command, then the training pipeline would get start.
```sh
sh scripts/train.sh
```
*Please refer to the trainft python scripts, and change accordingly in the train.sh scripts.*

To view training results and loss plots, run `visdom -port 8098` and click the URL http://localhost:8098.

This is a 3 stage training process.
1. We would start to train our full image colorization branch based on the [siggraph_retrained's pretrained weight](https://github.com/richzhang/colorization-pytorch).
2. We would use the full image colorization branch's weight as our instance colorization branch's pretrained weight.
3. Finally, we would train the fusion module.

## Testing the Instance-aware Image Colorization model
1. Our model's weight would place in [checkpoints/coco_mask](checkpoints/coco_mask).
2. Change the checkpoint's path in [test_fusion.py's L38](test_fusion.py#L38) from `coco_finetuned_mask_256_ffs` to `coco_mask`
3. Please follow the command below to colorize all the images in `example` foler based on the weight placed in `coco_mask`.

    ```
    python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
    ```
    All the colorized results would save in `results` folder.


## Acknowledgments
This project heavily utilises the code from https://github.com/ericsujw/InstColorization

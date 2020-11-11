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

## Getting Started (from original github)
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

## Pretrained Model 
1. Download it from [google drive](https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh).
```sh
sh scripts/download_model.sh
```
2. Now the pretrained models would place in [checkpoints](checkpoints).

To download the pretrained models fine-tuned for comic book dataset, please refer to:

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `example` folder.
```
python inference_bbox.py --test_img_dir example
```
All the prediction results would save in `example_bbox` folder.

## Colorize Images
Please follow the command below to colorize all the images in `example` foler.
```
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
```
All the colorized results would save in `results` folder.

* Note: all the images would convert into L channel to colorize in [test_fusion.py's L51](test_fusion.py#L51)

## Training the Model
Please follow this [tutorial](README_TRAIN.md) to train the colorization model.

This model can be adjusted for fine-tuning, please refer to the set of finetuning training python code. trainft.py, trainft_instance.py and trainft_fusion.py 



## Acknowledgments
This project heavily utilises the code from https://github.com/ericsujw/InstColorization

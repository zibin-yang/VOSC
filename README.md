# VOSC

 Usage
please follow the instructions to profiling authors.

## Dataset Preparation
```
usage: prepare_dataset.py [-h] [-i INPUT] [-o OUTPUT] [-s SIZE] [-l LENGTH]

optional arguments:
  -i INPUT      path to input dataset(default = './train/outdoor')
  -o OUTPUT     path to output directory(default = 'prepared_dataset')
  -s SIZE       number of videos in the desired dataset(default = 10)
  -l LENGTH     number of images in each video(default = 5)
```
running the script usage:
```
python3 prepare_dataset.py -i path_to_dataset_scenario_dir 
```

## FPN segmentation
```
usage: python fpn_segmentation.py 
```
running the script to train

If you are using cuda, pay attention to change the 'DEVICE' in the code and employ more workers in the dataloader of training dataset. The details of training time will be covered in Section 7.2 in the report. 

## Benchmark
I have not set up a benchmark myself. But I find something very interesting and we might utilize the [tool](https://github.com/open-mmlab/mmsegmentation).

## Data Loader
The notebook Image_loader.ipynb offers an image load helper under the 'Dataset' Class in pytorch. It provides some functions such as flip, rotate, resize, random crop, etc. Also, we can attach more functions on that by utilize the [albumentations package](https://github.com/albumentations-team/albumentations)

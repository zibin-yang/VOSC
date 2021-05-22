# VOSC

 Usage
please follow the instructions to profiling authors.

## Dataset Preparation`
Parameters:
```
usage: prepare_dataset.py [-h] [-i INPUT] [-o OUTPUT]

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

import os  
import re
import random
import argparse
import shutil

def mkdir(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.mkdir(path)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help="path to input dataset", default="./dataset/train/室内")
	parser.add_argument('-o', '--output', help="path to output directory", default='train_dataset')
	parser.add_argument('-s', '--size', help="size of data sample senarios", default=20)
	parser.add_argument('-l', '--length', help="size of images in each data sample", default=10)
	args = parser.parse_args()
	return args

def main():
	args = get_args()
	input_path = args.input
	output_path = args.output
	num_train = int(args.size)
	num_image = int(args.length)

	input_dir = os.path.normpath(input_path)
	out = os.path.normpath(output_path)
	mkdir(out)

	train_list = []
	count = 0
	for dir in os.listdir(input_dir):
		train_list.append(dir)
		count += 1

	sample_list = random.sample(train_list, num_train)
	out_image = os.path.join(out, 'image')
	out_mask = os.path.join(out, 'mask')
	mkdir(out_image)
	mkdir(out_mask)

	for dir in sample_list:
		train_element = os.path.join(input_dir, dir)
		files = os.listdir(train_element)
		ids = []
		for file in files:
			if (file.split('.')[1] == "jpg"):
				ids.append(file.split('.')[0])
		sample_images = random.sample(ids, num_image)
		for id in sample_images:
			shutil.copyfile(os.path.join(train_element, id + ".jpg"), os.path.join(out_image, id + ".jpg"))
			shutil.copyfile(os.path.join(train_element, id + ".png"), os.path.join(out_mask, id + ".png"))

	print("Dataset saved to " + str(out))

main()
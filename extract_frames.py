#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import shutil
from glob import glob
import video_utils
     
parser = argparse.ArgumentParser(description="""build a "frame dataset" from a given video""")
parser.add_argument('-video', '--input-video', dest="input_video", help='input video', required=True)
parser.add_argument('-name', '--dataset-name', dest="dataset_name", help='dataset name', required=True)
parser.add_argument('-p2pdir', '--pix2pix-dir', dest="pix2pix_dir", help='pix2pix directory', required=True)
parser.add_argument('-width', '--width', help='output width', default=1280, type=int)
parser.add_argument('-height', '--height', help='output height', default=736, type=int)
parser.add_argument('-fps', '--fps', help='fps', default=None, type=int)
args = parser.parse_args()

if not os.path.isfile(args.input_video):
    raise Exception("video does not exist")

if not os.path.isdir(args.pix2pix_dir):
    raise Exception("pix2pix directory does not exist")

if (args.width % 32 !=0) or (args.height % 32 !=0):
	raise Exception("please use width and height values that are divisible by 32")

print("creating the dataset structure")
dataset_dir = os.path.realpath(args.pix2pix_dir) + '/datasets/' + args.dataset_name
os.mkdir(dataset_dir)
os.mkdir(dataset_dir + "/train_frames")
os.mkdir(dataset_dir + "/test_frames")

video_utils.extract_frames_from_video(
	os.path.realpath(args.input_video),
	dataset_dir + "/train_frames",
	output_shape=(args.width, args.height),
	fps=args.fps
)

# copy first few frames to, for example, start the generated videos
for frame in sorted(glob(dataset_dir + "/train_frames/*.jpg"))[:60]:
    shutil.copy(
        frame,
        dataset_dir + "/test_frames"
    )


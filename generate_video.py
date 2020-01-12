### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

from tqdm import tqdm
from PIL import Image
import torch
import shutil
import video_utils
import image_transforms

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

# loading initial frames from: ./datasets/NAME/test_frames
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# this directory will contain the frames to build the video
frame_dir = os.path.join(opt.checkpoints_dir, opt.name, 'frames')
if os.path.isdir(frame_dir):
    shutil.rmtree(frame_dir)
os.mkdir(frame_dir)

frame_index = 1

if opt.start_from == "noise":
    # careful, default value is 1024x512
    t = torch.rand(1, 3, opt.fineSize, opt.loadSize)

elif opt.start_from  == "video":
    # use initial frames from the dataset
    for data in dataset:
        t = data['left_frame']
        video_utils.save_tensor(
            t,
            frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
            text="original video",
        )
        frame_index += 1
else:
    # use specified image
    filepath = opt.start_from
    print(filepath)
    if os.path.isfile(filepath):
        print('so far so good')
        t = video_utils.im2tensor(Image.open(filepath))
        for i in range(50):
            video_utils.save_tensor(
                t,
                frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
            )
            frame_index += 1
    else:
        print('oops, not a file using filepath')

current_frame = t

duration_s = opt.how_many / opt.fps
video_id = "epoch-%s_%s_%.1f-s_%.1f-fps%s" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps,
    "_with-%d-zoom" % opt.zoom_lvl if opt.zoom_lvl!=0 else ""
)

model = create_model(opt)

for i in tqdm(range(opt.how_many)):
    next_frame = video_utils.next_frame_prediction(model, current_frame)

    if opt.zoom_lvl != 0:
        next_frame = image_transforms.zoom_in(next_frame, zoom_level=opt.zoom_lvl)

    if opt.heat_seeking_lvl != 0:
        next_frame = image_transforms.heat_seeking(next_frame, translation_level=opt.heat_seeking_lvl, zoom_level=opt.heat_seeking_lvl)

    video_utils.save_tensor(
        next_frame, 
        frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
    )
    current_frame = next_frame
    frame_index+=1

video_path = output_dir + "/" + video_id + ".mp4"
while os.path.isfile(video_path):
    video_path = video_path[:-4] + "-.mp4"

video_utils.video_from_frame_directory(
    frame_dir, 
    video_path, 
    framerate=opt.fps, 
    crop_to_720p=True,
    reverse=False
)

print("video ready:\n%s" % video_path)

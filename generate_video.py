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

if opt.png:
    ext = 'png'
else:
    ext = 'jpg'

frame_index = 1
options_text = ""
if opt.start_from == "noise":
    # careful, default value is 1024x512
    t = torch.rand(1, 3, opt.fineSize, opt.loadSize)

elif opt.start_from  == "video":
    # use initial frames from the dataset
    for data in dataset:
        t = data['left_frame']
        video_utils.save_tensor(
            t,
            frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
            text="original video",
        )
        frame_index += 1
else:
    # use specified image
    filepath = opt.start_from
    options_text += ("_" + filepath.split('/')[-1])
    
    if os.path.isfile(filepath):
        t = video_utils.im2tensor(Image.open(filepath))
        
        if(opt.start_frames_before > 0):
            split = filepath.split('-')
            frame_no = split[-1]
            frame_n = int(frame_no.split('.')[0])
        
        for i in range(opt.start_frames_before):
            
            if(opt.start_frames_before > 0):
                f = frame_n - (opt.start_frames_before-i)
                f = str(f).zfill(6)+'.jpg'
                fpath = filepath.replace((frame_no),f)
                t = video_utils.im2tensor(Image.open(fpath))

            video_utils.save_tensor(
                t,
                frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
            )
            frame_index += 1
    else:
        print('oops, not a file using filepath')

current_frame = t

duration_s = opt.how_many / opt.fps
if (opt.zoom_lvl!=0):
   options_text += ("_with-%d-zoom" % opt.zoom_lvl) 

if(opt.zoom_cres):
    zoom_amt = 0
    options_text += ("_zoom-cres-%d" % opt.zoom_inc)

video_id = "epoch-%s_%s_%.1f-s_%.1f-fps%s" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps,
    options_text
)

model = create_model(opt)

for i in tqdm(range(opt.how_many)):
    next_frame = video_utils.next_frame_prediction(model, current_frame)

    if opt.zoom_lvl != 0:
        next_frame = image_transforms.zoom_in(next_frame, zoom_level=opt.zoom_lvl)

    if(opt.zoom_cres):
        if(i % opt.zoom_inc == 0):
            zoom_amt += 1

        next_frame = image_transforms.zoom_in(next_frame, zoom_level=zoom_amt)


    if opt.heat_seeking_lvl != 0:
        next_frame = image_transforms.heat_seeking(next_frame, translation_level=opt.heat_seeking_lvl, zoom_level=opt.heat_seeking_lvl)

    video_utils.save_tensor(
        next_frame, 
        frame_dir + "/frame-%s.%s" % (str(frame_index).zfill(5), ext),
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
    crop_to_720p=opt.no_crop,
    reverse=False,
)

print("video ready:\n%s" % video_path)

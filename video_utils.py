import util.util as util
import torch
import torchvision.transforms as transforms
import subprocess
import shlex

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def save_tensor(tensor, path, text="", text_pos="auto", text_color=(255,255,255)):
    """Saving a Torch image tensor into an image (with text)"""
    img_nda = util.tensor2im(tensor.data[0])
    img_pil = Image.fromarray(img_nda)

    if text != "":
        if text_pos == "auto":
            # top-right corner
            text_xpos = img_pil.width - 28 * len(text)
            text_ypos = 30
        else:
            text_xpos, text_ypos = text_pos

        draw = ImageDraw.Draw(img_pil)
#         font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 50)
#         draw.text((text_xpos, text_ypos), text, text_color, font=font)

    img_pil.save(path)

def save_img(img_pil, path, text="", text_pos="auto", text_color=(255,255,255)):
    if text != "":
        if text_pos == "auto":
            # top-right corner
            text_xpos = img_pil.width - 28 * len(text)
            text_ypos = 30
        else:
            text_xpos, text_ypos = text_pos

        draw = ImageDraw.Draw(img_pil)
#         font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 50)
#         draw.text((text_xpos, text_ypos), text, text_color, font=font)

    img_pil.save(path)


def im2tensor(img_pil):
    """Go from a PIL image (0..255 RGB) to a (-1..1) tensor"""
    transform_list = [
        # go from 0..255 to 0..1
        transforms.ToTensor(),
        # standard scaling: (t-0.5)/0.5 for all channels
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    t = transform(img_pil)
    return t.reshape((1, t.shape[0], t.shape[1], t.shape[2]))


def next_frame_prediction(generator, input_tensor):
    """Just one forward pass through the generator"""
    output_tensor = generator.inference(input_tensor, None, None)
    return output_tensor

def extract_frames_from_video(video_path, frame_dir, output_shape=(1280, 736),fps=None, ffmpeg_verbosity=16,png=False):
    """Extract all frames from a video
      - scale down the frames to match the desired height
      - crop to the desired width
    ex: 1920x1080 --> 1308x736 --> 1280x736
    """

    if png:
        ext = 'png'
    else:
        ext = 'jpg'

    width, height = output_shape
    if(fps is None):
        command = """ffmpeg -v %d -i %s -q:v 2 -vf "scale=iw*%d/ih:%d, crop=%d:%d" %s/frame-%%06d.%s -hide_banner""" % (
            ffmpeg_verbosity,
            video_path,
            height,
            height,
            width,
            height,
            frame_dir.replace(" ", "\\ "),
            ext
        )
    else:
        command = """ffmpeg -v %d -i %s -r %d -q:v 2 -vf "scale=iw*%d/ih:%d, crop=%d:%d" %s/frame-%%06d.%s -hide_banner""" % (
            ffmpeg_verbosity,
            video_path,
            fps,
            height,
            height,
            width,
            height,
            frame_dir.replace(" ", "\\ "),
            ext
        )
    print(command)
    print("extracting the frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

def video_from_frame_directory(frame_dir, video_path, frame_file_glob=r"frame-%05d.jpg", framerate=24, ffmpeg_verbosity=16, crop_to_720p=True, reverse=False):
    """Build a mp4 video from a directory frames
        note: crop_to_720p crops the top of 1280x736 images to get them to 1280x720
    """
    command = """ffmpeg -v %d -framerate %d -i %s -ss 1 -q:v 2%s %s%s""" % (
        ffmpeg_verbosity,
        framerate,
        frame_dir + "/" + frame_file_glob,
        ' -filter:v "crop=1280:720:0:16"' if crop_to_720p else "",
        "-vf reverse " if reverse else "",
        video_path
    )
    print(command)
    print("building video from frames")
    p = subprocess.Popen(shlex.split(command), shell=False)
    p.communicate()

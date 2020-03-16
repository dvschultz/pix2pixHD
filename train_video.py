### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torch
from torch.autograd import Variable

import os
import numpy as np
import shutil
import video_utils

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

  
if opt.debug:
    #opt.display_freq = 1
    #opt.print_freq = 1
    #opt.niter = 1
    #opt.niter_decay = 0
    #opt.max_dataset_size = 10
    pass

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True

# could be changed, but not tested
opt.resize_or_crop = "none"
opt.batchSize = 1

# this debug directory will contain input/output frame pairs
opt.print_freq = lcm(opt.print_freq, opt.batchSize) 
if opt.debug:
    debug_dir = os.path.join(opt.checkpoints_dir, opt.name, 'debug')
    if os.path.isdir(debug_dir):
        shutil.rmtree(debug_dir)
    os.mkdir(debug_dir)

if opt.scheduled_sampling:
    if opt.batchSize > 1:
        raise Exception('(for now) in "scheduled sampling" mode, --batchSize has to be 1')
    if not opt.serial_batches:
        raise Exception('(for now) in "scheduled sampling" mode, the --serial_batches option is necessary')
    if not opt.no_flip:
        raise Exception('(for now) in "scheduled sampling" mode, the --no_flip option is necessary')
    latest_generated_frame = None
    recursion = 0

opt.video_mode = True
data_loader = CreateDataLoader(opt) #this will look for a "frame dataset" at the location you specified
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
total_steps = (start_epoch-1) * dataset_size + epoch_iter
model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward Pass - frame t -> frame t+1 ######################

        if opt.scheduled_sampling and (latest_generated_frame is not None) and np.random.randn(1) < opt.ss_recursion_prob:
            left_frame = latest_generated_frame.detach()
            recursion += 1
        else:
            left_frame = Variable(data['left_frame'])
            recursion = 0

        right_frame = Variable(data['right_frame'])

        if opt.debug:
            video_utils.save_tensor(left_frame, debug_dir + "/step-%d-left-r%d.jpg" % (total_steps, recursion))
            video_utils.save_tensor(right_frame, debug_dir + "/step-%d-right.jpg" % total_steps)

        losses, latest_generated_frame = model(
            left_frame, None,
            right_frame, None,
            infer=opt.scheduled_sampling
        )

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass - frame1->frame2 ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
        else:
            loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16: 
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()


        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

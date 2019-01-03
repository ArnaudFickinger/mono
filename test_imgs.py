import torch
import os
import sys
from torch.autograd import Variable
import numpy as np
from model.load_model import ModelLoader
from skimage import io
from skimage.transform import resize

model = ModelLoader()

input_height = 384
input_width  = 512

def test_simple(model):
    model.switch_to_eval()

    dirListing = os.listdir("./imgs_to_test")
    editFiles = []
    for item in dirListing:
        if ".jpg" in item:
            editFiles.append("./imgs_to_test/"+item)
        
    imgs = []
    original_size = []
    results = []

    for path in editFiles:
        img = np.float32(io.imread(path))/255.0
        img = img[:,:,:3] #ignore alpha
        original_size.append((img.shape[0], img.shape[1]))
        img = resize(img, (input_height, input_width), order = 1)
        imgs.append(img)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)
        if torch.cuda.is_available():
            input_images = Variable(input_img.cuda())
        else:
            input_images = Variable(input_img)
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        pred_inv_depth = 1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
        results.append(pred_inv_depth)

    for i, im in enumerate(results):
        io.imsave('./results/'+str(i)+'.png', im)

    for i, im in enumerate(results):
        im_res = resize(im, original_size[i], order = 1)
        io.imsave('./results/'+str(i)+'_resized_to_original.png', im_res)

    res = np.vstack(results)
    im = np.vstack(imgs)

    res = np.stack([res,res,res], axis=2)
    io.imsave('./results/all_imgs.png', np.hstack([im, res]))

    sys.exit()

test_simple(model)





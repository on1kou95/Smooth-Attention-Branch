use_gpu = True
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
from metrics import LogNLLLoss
import cv2
import torch
from SAB.lib.models.utils import *
from utils import JointTransform2D, ImageToImage2D, Image2D
imgchant = 1

# crop = (args.crop, args.crop)

crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(r'./Train', tf_train)
val_dataset = ImageToImage2D(r'./Validation', tf_val)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)


device = torch.device("cuda")

from SAB.models.sab_m import SAB
model = SAB(img_size = 128, num_classes=2, imgchan = imgchant)
model.load_state_dict(torch.load(r'.\Output\best_modeldemo.pth'))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)
print(model)

criterion = LogNLLLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                             weight_decay=1e-5)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)

# for epoch in range(200):
#
#     epoch_running_loss = 0
#
#     for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
#         X_batch = Variable(X_batch.to(device='cuda'))
#         # print('X_batch',X_batch.shape)
#         y_batch = Variable(y_batch.to(device='cuda'))
#         # print('y_batch',y_batch.shape)
#
#         # ===================forward=====================
#
#         output = model(X_batch)
#
#         tmp2 = y_batch.detach().cpu().numpy()
#         # print('tmp2',tmp2.shape)
#         tmp = output.detach().cpu().numpy()
#         # print('tmp',tmp.shape)
#         tmp[tmp >= 0.5] = 1
#         # print('tmp',tmp.shape)
#         tmp[tmp < 0.5] = 0
#         # print('tmp',tmp.shape)
#         tmp2[tmp2 > 0] = 1
#         # print('tmp2',tmp2.shape)
#         tmp2[tmp2 <= 0] = 0
#         # print('tmp2',tmp2.shape)
#         tmp2 = tmp2.astype(int)
#         # print('tmp2',tmp2.shape)
#         tmp = tmp.astype(int)
#         # print('tmp',tmp.shape)
#
#         yHaT = tmp
#         # print('yHaT',yHaT.shape)
#         yval = tmp2
#         # print('yval',yval.shape)
#
#         loss = criterion(output, y_batch)
#
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_running_loss += loss.item()
#         # if epoch == 0:
#         #
#         # if epoch
#
#         # ===================log========================
#     # print( epoch_running_loss / (batch_idx + 1))
#     recent_loss = epoch_running_loss / (batch_idx + 1)
#     if epoch == 0:
#         best_loss = recent_loss
#     else:
#         if best_loss > recent_loss:
#             best_loss = recent_loss
#             torch.save(model.state_dict(), r'./Output/' + "best_model.pth")
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch, 200, recent_loss))
#     best_loss = best_loss
#     print("best_loss:",best_loss)
#
#     # if epoch == 10:
#     #     for param in model.parameters():
#     #         param.requires_grad = True
#     # if (epoch % 10) == 0:
#     #
#     #     for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
#     #         # print(batch_idx)
#     #         if isinstance(rest[0][0], str):
#     #             image_filename = rest[0][0]
#     #         else:
#     #             image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
#     #
#     #         X_batch = Variable(X_batch.to(device='cuda'))
#     #         y_batch = Variable(y_batch.to(device='cuda'))
#     #         # print("X_batch",X_batch.shape)
#     #         # print("y_batch",y_batch.shape)
#     #         # start = timeit.default_timer()
#     #         y_out = model(X_batch)
#     #         # print("y_out",y_out.shape)
#     #         # stop = timeit.default_timer()
#     #         # print('Time: ', stop - start)
#     #         tmp2 = y_batch.detach().cpu().numpy()
#     #         # print("tmp2",tmp2.shape)
#     #         tmp = y_out.detach().cpu().numpy()
#     #         tmp[tmp >= 0.5] = 1
#     #         tmp[tmp < 0.5] = 0
#     #         tmp2[tmp2 > 0] = 1
#     #         tmp2[tmp2 <= 0] = 0
#     #         tmp2 = tmp2.astype(int)
#     #         tmp = tmp.astype(int)
#     #
#     #         # print(np.unique(tmp2))
#     #         yHaT = tmp
#     #         yval = tmp2
#     #
#     #         epsilon = 1e-20
#     #
#     #         # del X_batch, y_batch, tmp, tmp2, y_out
#     #
#     #         yHaT[yHaT == 1] = 255
#     #         yval[yval == 1] = 255
#     #         fulldir = 'D:/project/Medical-Transformer/Output2' + "/{}/".format(epoch)
#     #         print(fulldir+image_filename)
#     #         if not os.path.isdir(fulldir):
#     #             os.makedirs(fulldir)
#     #         # print(X_batch.shape)
#     #         # print(yHaT.shape)
#     #         # print(yval.shape)
#     #         import nibabel as nib
#     #         import matplotlib.pyplot as plt
#     #
#     #         X_v = X_batch.detach().cpu().numpy()
#     #
#     #         cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])
#     #         # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
#     # fulldir = 'D:/project/Medical-Transformer/Output2' + "/{}/".format(epoch)
#     # # torch.save(model.state_dict(), fulldir + 'gatedaxialunet' + ".pth")
#     # torch.save(model.state_dict(), 'D:/project/Medical-Transformer/Output2' + "final_model.pth")


model.load_state_dict(torch.load(r'.\Output\best_modeldemo.pth'))
GT_mask = []
predicted_mask = []
for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    # print(batch_idx)
    if isinstance(rest[0][0], str):
        image_filename = rest[0][0]
    else:
        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
    print(batch_idx)
    X_batch = Variable(X_batch.to(device='cuda'))
    y_batch = Variable(y_batch.to(device='cuda'))

    y_out = model(X_batch)

    tmp2 = y_batch.detach().cpu().numpy()
    tmp = y_out.detach().cpu().numpy()

    tmp[tmp >= 0.5] = 1
    tmp[tmp < 0.5] = 0
    tmp2[tmp2 > 0] = 1
    tmp2[tmp2 <= 0] = 0
    tmp2 = tmp2.astype(int)
    tmp = tmp.astype(int)

    # print(np.unique(tmp2))
    yHaT = tmp
    yval = tmp2

    epsilon = 1e-20

    del X_batch, y_batch, tmp, tmp2, y_out

    yHaT[yHaT == 1] = 255
    yval[yval == 1] = 255

    print('yval:',yval.shape)
    print('yHaT:', yHaT.shape)
    if batch_idx == 0:
        predicted_mask = yHaT
        GT_mask = np.expand_dims(yval, axis=0)
    else:
        predicted_mask = np.row_stack((predicted_mask, yHaT))

        yval = np.expand_dims(yval, axis=0)
        GT_mask = np.row_stack((GT_mask, yval))
    fulldir = './Result' + "/"

    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)

    cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])

import nibabel as nib
THRESHOLD = 0.2

predicted_mask = predicted_mask.transpose([0, 2, 3, 1])
predicted_mask = predicted_mask[:,:,:,1]
predicted_mask = np.expand_dims(predicted_mask, axis=3)
img = nib.Nifti1Image(predicted_mask, np.eye(4))
img.to_filename(os.path.join(r"", 'predicted_mask.nii'))


GT_mask = (GT_mask.transpose([0, 2, 3, 1])>THRESHOLD)*1
img = nib.Nifti1Image(GT_mask, np.eye(4))
img.to_filename(os.path.join(r"", 'GT_mask.nii'))

import numpy as np
import nibabel as nib
from duibi import *


segment1=predicted_mask
print(segment1.shape)

gt=GT_mask
print(gt.shape)

aList = [];
Haudorff_Distance_ = hd95(segment1, gt)
aList.append(Haudorff_Distance_);
print("Haudorff_Distance : ", np.mean(aList))

aList = [];
TPR = TPR(segment1, gt)
aList.append(TPR);
print("TPR : ", np.mean(aList))

aList = [];
TNR = TNR(segment1, gt)
aList.append(TNR);
print("TNR : ", np.mean(aList))

aList = [];
dc = dc(segment1,gt)
aList.append( dc );
print( "dc : ",np.mean(aList))
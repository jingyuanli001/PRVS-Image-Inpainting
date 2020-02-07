from skimage.measure import compare_ssim
from skimage.color import rgb2gray
import numpy as np
import cv2
import skimage


def test_quality():
   # real_path = "../../Dataset//Pairs_street_view//paris_arged_eval//"
    real_path = "resized//"
    fake_path = "snapshots//default//images//result_final//"
    ssim_scores = []
    psnr_scores = []
    l1_losses = []

    for i in range(2000):
        fake = fake_path + "img_{:d}.png".format(i+1)
      #  real = real_path + "img_" + "0000{:d}".format(1+i)[-4:] + ".png"
        real = real_path+"{:d}.png".format(182638+i)

        imageA = rgb2gray(cv2.imread(fake)/255)
        imageB = rgb2gray(cv2.resize(cv2.imread(real), (256,256))/255)


        score = compare_ssim(imageA, imageB, win_size=51, data_range = 1)
        ssim_scores.append(score)
        psnr_scores.append(psnr1(imageA, imageB))
        l1_losses.append(L1loss(imageA, imageB))
    
    print("SSIM score is:", sum(ssim_scores)/len(ssim_scores))
    print("PSNR score is:", sum(psnr_scores)/len(psnr_scores))
    print("L1 Losses score is:", sum(l1_losses)/len(l1_losses))
    
    
def psnr1(img1, img2):
   return skimage.measure.compare_psnr(img1, img2, data_range = 1)

def L1loss(img1, img2):
    return np.sum(np.abs(img1 - img2))/np.sum(img1 + img2)

test_quality()
# Progressive Reconstruction of Visual Structure for Image Inpainting
## Requirements
Python >= 3.5

PyTorch >= 1.0.0

Opencv2 ==3.4.1

Scipy == 1.1.0

Numpy == 1.14.3

Scikit-image (skimage) == 0.13.1

This is the environment for our experiments. Later versions of these packages might need a few modifications of the code and it could lead to a decay of the performance (We are still checking this).

## Running the program
To perform training or testing, use 
```
python run.py
```
There are several arguments that can be used, which are
```
--data_root +str #where to get the images for training/testing
--mask_root +str #where to get the masks for training/testing
--model_save_path +str #where to save the model during training
--result_save_path +str #where to save the inpainting results during testing
--g_path +str #the pretrained generator to use during training/testing
--d_path +str #the pretrained discriminator to use during training/testing
--target_size +int #the size of images and masks
--mask_mode +int #which kind of mask to be used, 0 for external masks with random order, 1 for randomly generated masks, 2 for external masks with fixed order
--batch_size +int #the size of mini-batch for training
--n_threads +int
--gpu_id +int #which gpu to use
--finetune #to finetune the model during training
--test #test the model
```
For example, to train the network using gpu 1, with pretrained models
```
python run.py --data_root data --mask_root mask --g_path checkpoints/g_10000.pth --d_path checkpoints/d_10000.pth --batch_size 6 --gpu 1
```
to test the network
```
python run.py --data_root data --mask_root mask --g_path checkpoints/g_10000.pth --test --mask_mode 2
```
## Training procedure
To fully exploit the performance of the network, we suggest to use the following training procedure, in specific
1. Train the network, i.e. use the command
```
python run.py
```
2. Finetune the network, i.e. use the command
```
python run.py --finetune --g_path path-to-trained-generator --d_path path-to-trained-discriminator
```
3. Test the model
```
python run.py --test
```
## Building your own method
To modify the method or build your own method based on this code, you can do this by changing the PRVSNet.py and model.py files.

For example, to change the training target for generator, you can modify the get_g_loss method in model.py.

## Improving the code
Since this code is a rewrite version of our original experiment code, we haven't tested the thoroughly and there might exist some bugs. We will keep improving this code.
## Citation
@InProceedings{Li_2019_ICCV,

author = {Li, Jingyuan and He, Fengxiang and Zhang, Lefei and Du, Bo and Tao, Dacheng},

title = {Progressive Reconstruction of Visual Structure for Image Inpainting},

booktitle = {The IEEE International Conference on Computer Vision (ICCV)},

month = {October},

year = {2019}

} 

## Paper
To get the original copy of our paper, please go to:

http://openaccess.thecvf.com/content_ICCV_2019/html/Li_Progressive_Reconstruction_of_Visual_Structure_for_Image_Inpainting_ICCV_2019_paper.html
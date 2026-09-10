# SWINRGF
code for PBVS@CVPR 2026 Thermal Images Super-resolution challenge-Track2
# Set up 
```
conda create -n swin python=3.10
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
# Training script
If you need to train x8 images, run the code:
```
python train_SwinRGF.py --opt exp/train_x8.json
```
If you need to train x16 images, run the code:
```
python train_SwinRGF.py --opt exp/train_x16.json
```
# Testing script
If you need to test x8 images, run the code:
```
python test_SwinRGF.py --opt exp/test_x8.json
```
If you need to test x16 images, run the code:
```
python test_SwinRGF.py --opt exp/test_x16.json
```
# Pre-trained Model Weights
x8 pre-trained model weights: [x8_E.pth](https://pan.baidu.com/s/1oGkXQwqB0ZRIeOhKsRbWGg?pwd=vi4g )  
x16 pre-trained model weights: [x16_E.pth](https://pan.baidu.com/s/1OLZdex5AySNtpbnkQBqAGg?pwd=s2hs )
# Acknowledgement
Most of the code is based on the work of [SwinFuSR](https://github.com/VisionICLab/SwinFuSR) , modifying the `test_swinFuSR.py` code, and splitting the test scripts in the `options` folder into `test_swinFuSR_x8.json` and `test_swinFuSR_x16.json`.thanks to the team for their inspiration!

## Unsupervised-Segmentation
### An impermentation of Unsupervised Image Segmentation by Backpropagation  - Asako Kanezaki 金崎朝子 （東京大学）
#### **But faster and more elegant than origin version. (speed up from 30s to 5s)**

Paper: https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf
Original version Github: https://github.com/kanezaki/pytorch-unsupervised-segmentation

## Requement
Necessary: Python 3, Torch 0.4
Unnecessary: skimage, opencv-python(cv2)

## Getting Started
```
# Try the Code written by the original author
python3 demo_origin.py 
python3 demo_origin.py --input image/woof.jpg

# Try the high performance code written by me
python3 demo_modify.py

# Run this demo, and press WASDQE to adjust the parameters.
# The image show in the GUI, and the parameters show in terminal in real time.
# W,S --> parameter 1
# A,D --> parameter 2
# Q,E --> parameter 3
python3 demo_pre_seg__felz_slic.py
```

#### 如果你可以看的懂中文，那么我对这个算法的分析写在知乎上了，有兴趣就去看看吧（或者你是从知乎过来的）
#### 
#### If you can understand English, then I know you can understand this line of words.

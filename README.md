## Unsupervised-Segmentation
### An implementation of **Unsupervised Image Segmentation by Backpropagation  - Asako Kanezaki 金崎朝子** （東京大学）
### **Faster and more elegant than origin version. Speed up, 30s(origin) --> 5s(modify)**

Paper: https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf

Original version Github: https://github.com/kanezaki/pytorch-unsupervised-segmentation




## Requement

Necessary: Python 3, Torch 0.4

Unnecessary: skimage, opencv-python(cv2)




## Getting Started
Try the high performance code written by me.
```
python3 demo_modify.py

class Args(object):  # You can change the input_image_path ↓
    input_image_path = 'image/woof.jpg'  # image/coral.jpg image/tiger.jpg
```
  

Or you want to try the code written by the original author.
```
python3 demo_origin.py 
python3 demo_origin.py --input image/woof.jpg
```
  
Run this demo, and **press WASDQE on the keyboard** to adjust the parameters.
The image show in the GUI, and the parameters show in terminal in real time.
You could choose **Algorithm felz** or **Algorithm slic** by commenting the code.
* W,S --> parameter 1
* A,D --> parameter 2
* Q,E --> parameter 3
```
python3 demo_pre_seg__felz_slic.py
```


## Translate 翻译

#### If you can understand English, then I know you can understand this line of words (and you see this line on GitHub.)
#### 如果你可以看得懂中文，那么我对这个算法的分析写在知乎上了（或者你就是从知乎过来的）
  
  
#### An implementation of **Unsupervised Image Segmentation by Backpropagation**
#### 无监督图片语义分割，复现并魔改Github上的项目 https://zhuanlan.zhihu.com/p/68528056


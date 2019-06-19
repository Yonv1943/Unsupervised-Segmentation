## Unsupervised-Segmentation
### An implementation of **Unsupervised Image Segmentation by Backpropagation  - Asako Kanezaki 金崎朝子** （東京大学）ICASSP. 2018. 
### **Faster and more elegant than origin version. Speed up, 30s(origin) --> 5s(modify)**

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/ICASSP2018_modify.png "modify_title")


Paper: https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf

Original version Github: https://github.com/kanezaki/pytorch-unsupervised-segmentation

An Interpretation of this algorithm: https://zhuanlan.zhihu.com/p/68528056 (Warning: Simplified Chinese)


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

## Preview
The iterative process: Save the result when the iter_number == 1,2,4,8,16,32,64,128.

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/coral_128.gif "coral")

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/tiger_128.gif "tiger")

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/woof_128.gif "woof")
  


The different result of **Algorithm felz** or **Algorithm slic** with different parameters.

The left picture: compactness = 10000

The right picture: compactness = 1000

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/tiger_compactness.jpg "tiger_compactness")

The left picture: **Algorithm slic**

The right picture:  **Algorithm felz**

![](https://github.com/Yonv1943/Unsupervised-Segmentation/blob/master/readme_image/tiger_felz_slic.jpg "tiger_felz_slic")




## Translate 翻译

#### If you can understand English, then I know you can understand this line of words (and you see this line on GitHub.)
#### 如果你可以看得懂中文，那么我对这个算法的分析写在知乎上了（或者你就是从知乎过来的）
  
  
#### An implementation of **Unsupervised Image Segmentation by Backpropagation**
#### 无监督图片语义分割，复现并魔改Github上的项目 https://zhuanlan.zhihu.com/p/68528056


#### In my opinion, this algorithm is well suited for unsupervised segmentation of satellite images, because satellite images have no directionality. It is suitable for this algorithm with a priori assumption. (Priori Assumptions: In general, the regions with the same semantic information on the satellite images tend to occurs in a continuous area)
#### 这个算法很适合做 卫星图片的无监督语义分割任务，因为卫星地图没有方向性，并且地图上带有相同语义信息的区域往往是出现在一起的（符合先验假设）。很适合这种带有这种的先验假设算法。
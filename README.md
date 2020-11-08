# TECH NINJAS - Count and Classify the number of Kernels of Corn in an Image

## Introduction 

For variety of tasks in the field of Agriculture, classifying and counting the number of kernels or seeds of some particular plant is really necessary.

Be it the estimation of crop yield, the estimation of net profits a certain quality of crop can provide, the amount of nutrition which can be obtained, the quality of the food prepared from that particular quality of seeds, Counting and Classification of Seeds are a really important step

Here, for the sake of demonstration as a Proof of Concept as well as a Minimum Viable Product, We , the team at Tech Ninjas, present to you, a Seed Classifier and Counter on the basis of seeds or kernels of Corn or Maize or Cholam (as called in Malayalam)


## Problem Statement
In the field of Agriculture, Our Problem statements is defined as follows:

Come up with a product or application which can
- Successfully identify kernels of Corn from an Image
- Identify which all kernels are healthy or safe to be consumed
- Mark these kernels in our application and flag them as per their classification

## Work Flow 
The basic workflow of our Problem Statement can be differentiated into 4 major parts

- Data Collection and Labelling
- Masking and Extraction of seeds from an input Image
- Identifying and testing out several different types of  Convolutional Neural Network which can classify our seeds into their respective categories
- Analysing the different Neural Networks and comparing their accuracies and losses both in training and validation data
- Annotating the input images and flagging them as per their classification on our trained model
- Creating a flask endpoint for converting the Model into an API and associating a minimal FrontEnd Framework which can interact with our Server
- DEPLOYMENT


## STEP 1: 
We are using Convolutional Neural Networks and they are notorious for being Data Hungry Algorithms ie They need high amount of Data for generalizing the key features and providing with a relatively good accuracy

So we are going to scour for Open Source Data Hosting Platforms like Kaggle for the Data Sourcing Part due to time limitations as well as physical limitations

- [### THE TRAINING DATA CAN BE FOUND HERE](https://www.kaggle.com/zom8ie99/maize-seed)

### Credits : Kaggle and Dataset Owner [Manish Bhurtel](https://github.com/9characters)

- On Downloading the Files, we can observe that the dataset is structured as follows:

  ![DatasetTree](/UploadImages/DatasetTree.png)

## Step 2:
- Now we have to create a mask for our input image which can help us extract out our seeds from the image. The general approach towards this is first input the image through cv2 and then resize it and convert to GrayScale.

![OriginalImage](/UploadImages/OriginalImage.png)

```python
image = cv2.imread(//PATH OF INPUT IMAGE)
image = cv2.resize(image, (// RESIZE FACTOR))       //RESIZE FACTOR SHOULD BE FACTOR OF 16 LIKE 640 ETC
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
![GrayScale Image](/UploadImages/GrayscaleImage.png)

- Now as we have a resized grayscale image of our sample input, we can apply various masks to differentiate between the foreground and background 
```python
_, mask = cv2.threshold(gray_scale, 170, 255, cv2.THRESH_BINARY)
```
On visualizing the mask looks like this:

![Mask](/UploadImages/masknoerosion.png)

- We can observe that there are several noise spots in our mask, to reduce them, we can erode the mask with a kernel size of (5, 5)  (experimentally justified). Erosion canbe further learnt upon [here](https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/)
```python
kernel = np.ones((5, 5), np.uint8)
eroded_mask = cv2.erode(mask, kernel)
```

![Eroded Mask](/UploadImages/eroded_mask.png)

- From this contours can be extracted using OpenCvs findContours function and bounding rects can be estabilished for theindividual seeds. Ultimately we can then store them in a temporary folder where individual photos will looks somewhat like this:

![img1](/UploadImages/1.jpg)  
![img1](/UploadImages/2.jpg)    
![img3](/UploadImages/3.jpg)

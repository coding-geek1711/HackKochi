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

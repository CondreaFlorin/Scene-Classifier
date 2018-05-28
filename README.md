# Scene-Classifier
Project for for Computer Vision optional course at FMI UNIBUC

# Task
The purpose of this project was to create a CNN able to classify scenes in images.

# Dataset

Initially, the dataset to be used in the training and testing of the Network was the Places2 dataset, with included over 20 GB of images with scenaries, with 10 million and 400 categories. Unfortunatly, the dateset download was very difficult, taking around 8 to 10 hours. In the end, a smaller and older dataset was used, indoorCVPR_09. which had only 2 GB of data, consisting in 14000 photos and 62 categories.

# Data Preprocessing

The data didn't have a standard size, so i reduced the images to 64 x 64, as presented in the paper as well, then i applied normalization the the images. To add more artificial data based on the originals, keras image Data Generator was used, doubling the amount of pictures, and given the nature of the problem, relabeling wasn't neccesary. 

# Inspiration

I read paper [Scene classification with Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2017/pdfs/102.pdf) from Stanford, and tried basing my project on it, but the arhitectures presented there were too big to be trained localy on GPU, and training on CPU would have taken a really long time.

# Arhitecture

I started from the baseline arhitecture presented in the paper mentioned before, but the results were really bad, accuracy of about 5% and the model overfitted the data in a very small number of epochs. Through trial and error, the performance of the network was improved by striping away layers, adding dropout after each conv2D, then adding batch normalization.

## Initial functional arhitecture
 
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 64)        9472      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 58, 58, 64)        200768    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 29, 29, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 29, 29, 64)        36928     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 27, 27, 64)        36928     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 27, 27, 128)       73856     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 25, 25, 128)       147584    
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 25, 25, 256)       295168    
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 23, 23, 256)       590080    
_________________________________________________________________
flatten_1 (Flatten)          (None, 135424)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 68)                9208900   
=================================================================
Total params: 10,599,684
Trainable params: 10,599,684
Non-trainable params: 0

## Best Arhitecture

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        9472      
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
batch_normalization_3 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 2, 2, 512)         1180160   
_________________________________________________________________
batch_normalization_4 (Batch (None, 2, 2, 512)         2048      
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 2, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 68)                139332    
=================================================================
Total params: 1,701,828
Trainable params: 1,699,908
Non-trainable params: 1,920

# Training hyperparameters

Epochs = 100
Batch Size = 128 ( smaller batch size led to under fitting)
Data split = (Train/test) (80%/20%)
Optimizer = Adam, initial RMSprop
Metrics = ['accuracy', 'top_k_categorical_accuracy']

# Training Progress

There have been 14 trains with different arhitectures and hyperparameters. Presenting only a couple more interesting, the plots for all are located in the Plots folder.

## Run nr 3

This was the first run to return results, but the model overfitted the data to an extreme degree, train accuracy being over 99%, while test being around 10%. Same thing happed in the next 3 runs, but to a lesse degree, indicating we were making progress in reducing overfitting

Test Loss , Test accuracy = [10.979912623035265, 0.12339331627198541]

![Loss](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/loss3.png)
![Acc](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/acc3.png)

## Run nr 7

The model is starting to no longer overfit the data, having added a weak dropout after each layer of 0.25, getting a few epochs with good results, and a test accuracy of about 20%.

![Loss](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/loss7.png)
![Acc](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/acc7.png)

## Run nr 10

The model has started to work properly, no longer overfitting the data, having added a dropout of 0.6 and artificial images created by the image generator.

Test loss and accuracy = [2.917946684249736, 0.2614546620341609]

![Loss](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/loss10.png)
![Acc](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/acc10.png)

## Run nr 11

Batch normalization has been added, which increased the accuracy hugely.
Test loss and accuracy = [2.556787643573268, 0.34658544411946657]

![Loss](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/loss11.png)
![Acc](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/acc11.png)

## Run nr 12

The best run of the arhitecture, having an Top1 acurracy of 36% and Top5 of 67%.
Test loss, accuracy, top k accuracy = [2.5326014738816482, 0.3682692307692308, 0.6782051282051282]

![Loss](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/loss12.png)
![Acc](https://github.com/CondreaFlorin/Scene-Classifier/blob/master/Plots/acc12.png)

# Future Work

Given that them model no longer overfits the data, more images and longer training should increase the accuracy of the model. At the same time, building a more complex network either by tweak the parameters, or by adding more layers, since the training fits the testing well, we can make a trade off there, adding a bit of overfitting in exchange for better accuracy.

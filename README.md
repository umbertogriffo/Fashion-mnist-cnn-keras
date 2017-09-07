# Zalando's article images Recognition using Convolutional Neural Networks in Python with Keras

* Author: Umberto Griffo
* Twitter: @UmbertoGriffo

## Environment and software requirements
	
	* 2 Intel Xeon E5-2630 v4 2.2GHz, 25M Cache, 8.0 GT/s QPI, Turbo, HT, 10C/20T (85W) Max Mem 2133MHz
	* 128 GB Ram
	* 1 TB Disk
	* Python 3.5, TensorFlow 1.2.0, Keras 2.0.5, numpy, matplotlib, pandas 0.20.2, scikit-learn 0.18.1, h5py 2.7.0

## Data
You can download the Fashion Mnist dataset from Kaggle <a href="https://www.kaggle.com/zalando-research/fashionmnist">here</a> and then put the files into **/Dataset** directory.
		
## Training
execute fashion_mnist_cnn.py

## Preprocessing
Normalization

## Cross Validation
5-fold cross-validation

## CNN configuration
The network topology can be summarized as follows:

    - Convolutional layer with 32 feature maps of size 5×5.
    - Pooling layer taking the max over 2*2 patches.
    - Convolutional layer with 64 feature maps of size 5×5.
    - Pooling layer taking the max over 2*2 patches.
    - Convolutional layer with 128 feature maps of size 1×1.
    - Pooling layer taking the max over 2*2 patches.
    - Flatten layer.
    - Fully connected layer with 1024 neurons and rectifier activation.
    - Dropout layer with a probability of 50%.
    - Fully connected layer with 510 neurons and rectifier activation.
    - Dropout layer with a probability of 50%.
    - Output layer.

## Results

**Accuracy scores:**  [0.92433, 0.92133, 0.923581, 0.92391, 0.92466]

**Mean Accuracy:** 0.923567

**Stdev Accuracy:** 0.001175

**Final Accuracy:** 92.56%

The following picture shows the trend of the Accuracy of the final learning: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Fashion-mnist-cnn-keras/blob/master/Output/model_accuracy_fm_cnn.png"/>
</p>
           
## References

- [1] Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist

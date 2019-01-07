# Zalando's article images Recognition using Convolutional Neural Networks in Python with Keras

* Author: Umberto Griffo
* Twitter: @UmbertoGriffo

## Software requirements
	
	* Python 3.6, TensorFlow 1.11.0, Keras 2.2.4, numpy, matplotlib, scikit-learn, h5py

## Training
execute **fashion_mnist_cnn.py**

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

I evaluated the model using the 5-fold cross-validation on 60,000 examples divided into train and test.

**Accuracy scores:**  [0.92433, 0.92133, 0.923581, 0.92391, 0.92466]

**Mean Accuracy:** 0.923567

**Stdev Accuracy:** 0.001175

I ran a new learning from scratch on 60,000 examples and then I evaluated test accuracy on the test set of 10,000 examples.

**Final Accuracy:** 92.56%

The following picture shows the trend of the Accuracy of the final learning: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Fashion-mnist-cnn-keras/blob/master/Output/model_accuracy_fm_cnn.png"/>
</p>
           
## References

- [1] Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist

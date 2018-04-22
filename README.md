Multilayer perceptron model is a class of artificial neural network that uses back propagation technique for training. It has three layers of nodes. The hidden layer has 4, the input layer has 3 and the output layer has 6 neurons. During training, all the weights and biases are updated after processing single training sample and the method is called stochastic training. The update rule uses learning rate of 0.1 and momentum factor of 0.001. The code is run for 1500 epochs.
The activation function is sigmoid of the form 
  f(x) = 1 / 1 + exp(-x) 
Its range is between 0 and 1. It is a S — shaped curve. 
The performance of classification is measured using confusion matrix. The code computes specificity, sensitivity, precision, recall, accuracy and F-score. The model uses 5-fold cross validation to improve the prediction accuracy and reduce variance. The dataset is divided into 5 folds wherein 4 subsamples are used to train the network and the 5th subsample tests the model. The process is repeated for each fold of the dataset. 





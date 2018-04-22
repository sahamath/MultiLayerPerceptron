Multilayer perceptron model is a class of artificial neural network that uses back propagation technique for training. It has three layers of nodes. The hidden layer has 4, the input layer has 3 and the output layer has 6 neurons. During training, all the weights and biases are updated after processing single training sample and the method is called stochastic training. The update rule uses learning rate of 0.1 and momentum factor of 0.001. The code is run for 1500 epochs.
The activation function is sigmoid of the form 
  f(x) = 1 / 1 + exp(-x) 
Its range is between 0 and 1. It is S — shaped curve. 
The performance of classification is measured using confusion matrix. The code computes specificity, sensitivity, precision, recall, accuracy and F-score. The model uses 5-fold cross validation to improve the prediction accuracy and to reduce variance. The dataset is divided into 5 folds wherein 4 subsamples are used to train the network and the 5th subsample tests the model. The process is repeated for each fold of the dataset. 
loadCsv function loads the input ‘data.csv’ file into local memory. Minmax computes the minimum and the maximum value of features for normalization. cross_validation_split divides the dataset into k folds of equal samples. Run_algorithm takes one fold at a time, considers the fold as test set and trains the network with samples of the remaining (merged) folds.
Back_propagation function initializes the network by assigning random weights to the connections of input, hidden and output layers. The train_network trains the network by first propagating the inputs through a system of weighted connections. For each input sample, the intermediate layer computes weighted sum, derives activation function value and forwards the output of the neuron to the next layer. The output layer compares the expected and the obtained neuron value and back propagates the difference so that the weight adjustments can minimise error for the next iteration. activate function computes the weighted sum of inputs, transfer function calculates neuron output due to the activation function, backward_propagate_error derives error at every layer using the delta rule, and update_weight stores the modified weights of neurons at each layer of the network.
Confusion_matrix uses the actual and predicted output to generate values for false positives, false negatives, true positives and true negatives.  
The output :
Fold 1:
9   1   0   0   3   4
1   8   0   0   0   0
0   0  20   0   7   0
0   0   0  27   2   2
4   0   5   0  38   0
 0   1   0   4   1  37
False Positives
 [ 5  2  5  4 13  6]
False Negetives
 [8 1 7 4 9 6]
True Positives
 [ 9  8 20 27 38 37]
True Negetives
 [152 163 142 139 114 125]
Sensitivity 
 [ 0.52941176  0.88888889  0.74074074  0.87096774  0.80851064  0.86046512]
Specificity 
 [ 0.96815287  0.98787879  0.96598639  0.97202797  0.8976378   0.95419847]
Precision 
 [ 0.64285714  0.8         0.8         0.87096774  0.74509804  0.86046512]
Recall 
 [ 0.52941176  0.88888889  0.74074074  0.87096774  0.80851064  0.86046512]
Áccuracy 
[ 0.92528736  0.98275862  0.93103448  0.95402299  0.87356322  0.93103448]
FScore 
[ 0.58064516  0.84210526  0.76923077  0.87096774  0.7755102   0.86046512]
Çohen Kappa 
0.7469458987783595

Fold 2
Confusion matrix –

   9   4   0   0   4   0
   1  19   0   0   0   1
   0   0  36   0   4   0
   0   0   0  24   0   4
   1   0   0   0  36   2
   0   1   0   1   1  26

False Positives
 [2 5 0 1 9 7]
False Negetives
 [8 2 4 4 3 3]
True Positives
 [ 9 19 36 24 36 26]
True Negetives
 [155 148 134 145 126 138]
Sensitivity 
 [ 0.52941176  0.9047619   0.9         0.85714286  0.92307692  0.89655172]
Specificity 
 [ 0.98726115  0.96732026  1.          0.99315068  0.93333333  0.95172414]
Precision 
 [ 0.81818182  0.79166667  1.          0.96        0.8         0.78787879]
Recall 
 [ 0.52941176  0.9047619   0.9         0.85714286  0.92307692  0.89655172]
Áccuracy 
[ 0.94252874  0.95977011  0.97701149  0.97126437  0.93103448  0.94252874]
FScore 
[ 0.64285714  0.84444444  0.94736842  0.90566038  0.85714286  0.83870968]
Çohen Kappa 
0.8311567541341527
   Fold 3
  Confusion matrix -

   6   2   0   0   4   0
   1  16   0   0   0   1
   0   0  35   0  10   0
   0   0   0  21   0   7
   3   0   3   0  31   0
   0   2   0   1   0  31

False Positives
 [ 4  4  3  1 14  8]
False Negetives
 [ 6  2 10  7  6  3]
True Positives
 [ 6 16 35 21 31 31]
True Negetives
 [158 152 126 145 123 132]
Sensitivity 
 [ 0.5         0.88888889  0.77777778  0.75        0.83783784  0.91176471]
Specificity 
 [ 0.97530864  0.97435897  0.97674419  0.99315068  0.89781022  0.94285714]
Precision 
 [ 0.6         0.8         0.92105263  0.95454545  0.68888889  0.79487179]
Recall 
 [ 0.5         0.88888889  0.77777778  0.75        0.83783784  0.91176471]
Áccuracy 
[ 0.94252874  0.96551724  0.92528736  0.95402299  0.88505747  0.93678161]
FScore 
[ 0.54545455  0.84210526  0.84337349  0.84        0.75609756  0.84931507]
Çohen Kappa 
0.7583234609256915

Fold 4
Confusion matrix -

   5   1   0   0   3   1
   1  21   0   0   0   0
   0   0  27   0   4   0
   1   0   0  30   1   3
   2   0   1   0  37   0
   0   0   0   0   2  34

False Positives
 [ 4  1  1  0 10  4]
False Negetives
 [5 1 4 5 3 2]
True Positives
 [ 5 21 27 30 37 34]
True Negetives
 [160 151 142 139 124 134]
Sensitivity 
 [ 0.5         0.95454545  0.87096774  0.85714286  0.925       0.94444444]
Specificity 
 [ 0.97560976  0.99342105  0.99300699  1.          0.92537313  0.97101449]
Precision 
 [ 0.55555556  0.95454545  0.96428571  1.          0.78723404  0.89473684]
Recall 
 [ 0.5         0.95454545  0.87096774  0.85714286  0.925       0.94444444]
Áccuracy 
[ 0.94827586  0.98850575  0.97126437  0.97126437  0.92528736  0.96551724]
FScore 
[ 0.52631579  0.95454545  0.91525424  0.92307692  0.85057471  0.91891892]
Çohen Kappa 
0.8581675904792958

Fold 5
Confusion matrix -

   7   4   0   0   5   0
   5  14   0   0   0   0
   0   0  22   0   7   0
   0   0   0  25   0   4
   1   0   1   0  39   2
   0   2   0   2   0  34
False Positives
 [ 6  6  1  2 12  6]
False Negetives
 [9 5 7 4 4 4]
True Positives
 [ 7 14 22 25 39 34]
True Negetives
 [152 149 144 143 119 130]
Sensitivity 
 [ 0.4375      0.73684211  0.75862069  0.86206897  0.90697674  0.89473684]
Specificity 
 [ 0.96202532  0.96129032  0.99310345  0.9862069   0.90839695  0.95588235]
Precision 
 [ 0.53846154  0.7         0.95652174  0.92592593  0.76470588  0.85      ]
Recall 
 [ 0.4375      0.73684211  0.75862069  0.86206897  0.90697674  0.89473684]
Áccuracy 
[ 0.9137931   0.93678161  0.95402299  0.96551724  0.90804598  0.94252874]
FScore 
[ 0.48275862  0.71794872  0.84615385  0.89285714  0.82978723  0.87179487]
Çohen Kappa 
0.7658715596330274


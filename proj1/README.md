# EECS738ml-KU
This the source code for project 1 of EECS 738 machine learning
Given by Dr. Martin Kuehnhausen
From (author) Ben Liu, Daksh Shukla

!!! please use Python 2.7.15 to run the program.

===============================================================================

Idea of model the distribution within the data:
According to texture book PRML(9), we can apply K-means to roughly cluster the
data points first, to save the cost of computing variances in EM, and then apply
EM to modelling Gaussian mixtures. Here we only focus on continuous feature.
1) we first normalize each feature by mapping original value to interval [0, 1], 
   then apply K-means to cluster the data-points.
2) starting from K = 2, the maximal number of repeats are set to be 100, K are
   set to be strictly smaller than [data_amount/13], to control running time.
   
   2.a) we randomly pick K data points as the centers.
   
   2.b) compute the distance of all data points to these centers.
   
   2.c) cluster all data points to the closest center, if there is any empty 
        cluster, go back to 2.a), otherwise continue.
        
   2.d) if the distance (defined as 2-norm) from new center to old center is
        smaller than epsilon (choose 1e-12), go to 2.e); otherwise go to 2.b).
        
   2.e) for each cluster, we compute the mean(mu) and standard-derivation(delta), 
        identify as outlier data point outside the interval [mean-3*delta, mean+3*delta].
        
   2.f) if the number of outliers is smaller than 13, stop; otherwise, increase
        K by 1 and go back to 2.b).
        
3) after 2) we have the number of cluster, K, and data points contained by each
   cluster. Now we are goint to model the mixture, using K Gaussians. 
   
   3.a) initialize Mu as the mean, Sigma as the Covariance matrix of each cluster,
        Pi as the ratio of number of data in each cluster, to total number of data.
        
   3.b) computing Gamma as PRML(9), this is the E-step.
   
   3.c) updating Mu, Sigma and Pi as PRML(9), this is the M-step.
   
   3.d) if the changes of Mu, Sigma and Pi are smaller than epsilon1 (choose 1e-5), stop,
        otherwise go back to 3.b).
        
4) Once we are done with the modelling, we will output the final parameters used to model
   the distribution within the data. We only visualize results when the number of features
   of interested is 1 or 2. If it is 1, we will output the hist and modelled pdf; if it is
   2, we will output the scatter of data points, before and after the clusterring.
   
===============================================================================

Call the program by:
python Model.py <file.csv> [<int.numer_of_features>]

<int.numer_of_features> is set to be 1 by default.


===============================================================================

Included files:

em_esl8_5.py        - a sample code to help us understanding EM from ISL, by Daksh Shukla.

Model.py            - main program to model the distribution, by Ben.

Iris.csv            - data set 1 from Kaggle

winequality-red.csv - data set 2 from Kaggle

winequality_clustering.pdf/txt

                    - result of modeling winequality-red.csv using 2 features. ~.txt showes 
                      the parameter for each Gaussian of the misture, ~.pdf shows the 
                      clusterring results.
                      
winequality_fitting.pdf/txt   

                    - result of modeling winequality-red.csv using 1 features. ~.txt showes 
                      the parameter for each Gaussian of the misture, ~.pdf shows the 
                      fitting results, noted that the program fixes the number of bins in hist,
                      so it may seem skewed.
                      
Iris_clustering.pdf/txt, Iris_fitting.pdf/txt

                    - similar results, but for Iris.csv
                    
===============================================================================

some discussion:
1) In practice, we cannot guarantee the data coming from a mixture, instead of one Gaussian, so 
   the K-means should start from K = 1. Here we make it 2 to force the program to model the 
   distribution using mixture. 
   On the other hand, from winequality_clustering.txt, which uses 6 Gaussians to model the 
   distribution, but Pi_0, Pi_2 and Pi_4 are less than 0.1, so they didnot play importance roles
   in the mixture compared to the other 3 Gaussians. So we think it's okay to start from K = 2.
2) since we randomly initialize the centers for K-means, the results might be different even for 
   the same data set. PRML did mention this is a problem for K-means.


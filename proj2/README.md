This the source code for project 1 of EECS 738 machine learning Given by Dr. Martin Kuehnhausen From (author) Ben Liu

===== Ideas ==============================================================
To describe a HMM, we need 5 parameters: 
  states, {0,1,...,N-1};
  alphabet, {'word1', 'word2', ... , 'wordm'};
  transition probability, A = {aij} = P(st+1=j|st=i), the probability that the HMM jump from state i to state j at time t;
  emission probability, B = {bij}, the state i will emit word j;
  initial state probability, P = {Pi}, the probability that the first state of the HMM is state i.
  
The only hyper-parameter is states, or number of states. Alphabet will be determined by the dataset, A, B and P are supposed to be trained from the training data.

We apply Baum-Welch algorithm to train the HMM, which is based on the forward-backward algorithm, to update A, B and P. 
Once the training is done, then we apply the Viterbi algorithm to estimate the probability that a given sequence is generated by the trained HMM. As the absolute probability will be affect by the length of given sequence. We will determine the given sequence coming from the HMM by:
  1) We shuffle the given sequence and them compute the so-called "background probability", repeat 100 times.
  2) We compute the mean and standard diveation of the 100 background probabilities, normalize it by (X-mean[X])/std[X].
  3) Use the same formula to "normalize" the estimate probability of the original given sequence.
  4) If the normalized estimate probability of the original given sequence is bigger than 3, it means the probability of the        given sequence coming from the HMM is sighificant higher than the background probabilities, so it will be regarded as          "true sequence" in the sense that it's generated from the HMM. We use 3 here since we normalized the distribution in 2). 

===== Included files ==============================================================
alllines.txt -- training data from https://www.kaggle.com/kingburrito666/shakespeare-plays
HMM_log.py   -- functions that is used by HMM, it should be noted that the forward-backward algorithm and Baum-Welch algorithm
                are scaled by taking logarithms of all probabilities (and their intermidiate value). But the Viterbi algorithm
                is still computing in the real probability domain.
train.py     -- the program to load data from alllines.txt and train HMM using functions in HMM_log.py, call it by:
                  python2 train.py
estimate.py  -- the program to estimate a given sequence is a "shakespeare" sentence or not. By default the program will use   
                "to be or not to be" as example sentences. But user can use other sentence by calling it:
                  python2 estimate.py <sentence>
                it should be noted that all words must be in the alllines.txt, and be lower case.
hmm_parameter.txt
             -- the file containing all pre-trained parameter since it takes a while to train HMM
                the first line is the number of states, N, used in the HMM;
                the seconde line is the alphabet of used in the HMM, so one can choose words from here to run estimate.py;
                the third line is the initial state probability, P;
                the fourth to 4+N lines are the transition probability matrix, A;
                the rest of lines are the emission probability matrix, B.

To make things easier, we also included a toy example, a occasionally dishonest casino:
The HMM is used to describe the process of rolling 6-side die, the dice could be either a fair dice or a loaded die. The loaded die has probability 0.5 of a '6' and probability 0.1 for the numbers '1' to '5'. If the casino is using the fair die, then the probability they are going to switch to loaded die is 0.05; 0.1 vice versa.

rollDice.py   -- the program to simulate die sequence from the described process above.
pure_seq.txt  -- one generated sequence of length 30,000
toy_train.py, toyhmm_parameter.txt, toy_estimate.py
              -- the same as before but for the toy example.
              
===== Discussion ==========================================================================
It's of great difficlty to choose the number of state for HMM, it takes efforts and experience to find a "best" parameter.
We need a better way to decouple the effect of sequence length on the estiamted probability.

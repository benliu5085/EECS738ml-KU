import numpy as np

"************ extended functions **********************************************"
def getV(cha, apl):
    return apl.index(cha)

def eexp(x):
    return np.exp(x)

def eln(x):
    return np.log(x)

def elnsum(x,y):
    if x > y:
        return x + eln(1 + eexp(y-x))
    else:
        return y + eln(1 + eexp(x-y))

def elnproduct(x, y):
    return x + y

def forardHMM(alphabet, O, A, B, P):
    N = len(P)
    K = range(0,N)  # states, 0, 1, ... , N-1
    T = len(O)      # number of observations
    ### 5. forward ####################################
    ELNA = np.zeros((T, N))
    for i in K:
        ELNA[0,i] = elnproduct(eln(P[i]) , eln(B[i, getV(O[0], alphabet)]))

    for t in range(1, T):
        for j in K:
            logalpha = -np.inf
            for i in K:
                logalpha = elnsum(logalpha, elnproduct(ELNA[t-1, i], eln(A[i,j])))
            ELNA[t,j] = elnproduct(logalpha,eln(B[j, getV(O[t], alphabet)]))

    return ELNA

def backwardHMM(alphabet, O, A, B, P):
    N = len(P)
    K = range(0,N)  # states, 0, 1, ... , N-1
    T = len(O)      # number of observations
    ### 6. backward ###################################
    ELNB = np.zeros((T, N))
    for t in np.arange(T-2,-1,-1):
        for i in K:
            logbeta = -np.inf
            for j in K:
                logbeta = elnsum(logbeta, elnproduct(eln(A[i,j]),
                                                     elnproduct(eln(B[j, getV(O[t+1], alphabet)]),
                                                                ELNB[t+1,j])))
            ELNB[t,i] = logbeta
    return ELNB

def trainHMM(N, alphabet, O, CC):
    """ Train HMM using Baum-Welch algorithm """
    K = range(0,N)  # states, 0, 1, ... , N-1
    T = len(O)      # number of observations
    A = np.zeros((len(K), len(K)))
    B = np.zeros((len(K), len(alphabet)))
    P = np.random.random_sample((len(K)))

    """random initialization"""
    P = P/P.sum()

    for i in range(0, A.shape[0]):
        t = np.random.random_sample((A.shape[1]))
        A[i,] = t/t.sum()

    for i in range(0, B.shape[0]):
        t = np.random.random_sample((B.shape[1]))
        B[i,] = t/t.sum()

    """forard and backward algorithm"""
    for cnt in range(0, CC):
        ELNA = forardHMM(alphabet, O, A, B, P)
        ELNB = backwardHMM(alphabet, O, A, B, P)
        ### 7. compute ELNR ###############################
        ELNR = np.zeros((T, N))
        for t in range(0, T):
            normalizer = -np.inf
            for i in K:
                ELNR[t,i] = elnproduct(ELNA[t,i], ELNB[t,i])
                normalizer = elnsum(normalizer, ELNR[t,i])
            for i in K:
                ELNR[t,i] = elnproduct(ELNR[t,i], -normalizer)

        ### 8. compute ELNS ###############################
        ELNS = np.zeros((T, N, N))
        for t in range(0, T-1):
            normalizer = -np.inf
            for i in K:
                for j in K:
                    ELNS[t,i,j] = elnproduct(ELNA[t,i],
                                             elnproduct(eln(A[i,j]),
                                                        elnproduct(eln(B[j, getV(O[t+1], alphabet)]), ELNB[t+1,j])
                                                       )
                                            )
                    normalizer = elnsum(normalizer, ELNS[t,i,j])

            for i in K:
                for j in K:
                    ELNS[t,i,j] = elnproduct(ELNS[t,i,j], -normalizer)

        ### 9. update P ###################################
        P = np.exp(ELNR[0,])

        ### 10.update A ###################################
        for i in K:
            for j in K:
                numerator = -np.inf
                denumerator = -np.inf
                for t in range(0,T-1):
                    numerator = elnsum(numerator, ELNS[t,i,j])
                    denumerator = elnsum(denumerator, ELNR[t,i])
                A[i,j] = eexp(elnproduct(numerator, -denumerator))

        ### 11.update B ###################################
        for j in K:
            for k in O:
                numerator = -np.inf
                denumerator = -np.inf
                for t in range(0,T):
                    if O[t] == k:
                        numerator = elnsum(numerator, ELNR[t,j])
                    denumerator = elnsum(denumerator, ELNR[t,j])
                B[j, getV(k, alphabet)] = eexp(elnproduct(numerator, -denumerator))

    return (A, B, P)

def viterbiHMM(alphabet, O, A, B, P):
    N = len(P)
    K = range(0,N)  # states, 0, 1, ... , N-1
    T = len(O)      # number of observations

    V = np.zeros((T, N))
    PTR = np.zeros((T, N))
    for i in K:
        V[0,i] = B[i, getV(O[0], alphabet)] * P[i]
        PTR[0,i] = i

    for t in range(1, T):
        for l in K:
            mk = -1
            maxima = -1
            for i in K:
                tl = V[t-1,i] * A[i,l]
                if tl > maxima:
                    maxima = tl
                    mk = i
            V[t,l] = B[l, getV(O[t], alphabet)] * maxima
            PTR[t,l] = mk

    return (V, PTR)

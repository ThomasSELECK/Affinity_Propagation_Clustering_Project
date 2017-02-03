import numpy as np

class Mapper:

    """
    This class defines a mapper object that computes the partial centroids for the partial dataset given in arguments.
    """
    
    def __init__(self, data):
        """
        This is the class' constructor.
        """
        
        # Dataset
        self.__data = data
        
        # Number of samples in the dataset
        self.__N = data.shape[0]
        
        # Availabilities matrix
        self.__A = np.zeros((self.__N, self.__N))
        
        # Responsabilities matrix
        self.__R = np.zeros((self.__N, self.__N))
        
        # Similarity matrix
        self.__S = np.zeros((self.__N, self.__N))
        
        # List of centroids
        self.__centers = []
    
    def __a(self, i, k):
        """
        This method computes the availability sent from point i to point k.
        
        Parameters
        ----------
        i : integer
                This is the index i needed to select the first point.

        k : integer
                This is the index k needed to select the first point.

        Returns
        -------
        a : float
                This is the availability of point i for k.
        """

        if i != k:
            a = min([0, self.__R[k, k] + sum([max(0, self.__R[i_prime, k]) for i_prime in range(self.__N) if i_prime != i and i_prime != k])])
        else:
            a = sum([max(0, self.__R[i_prime, k]) for i_prime in range(self.__N) if i_prime != k])
            
        return a

    def __r(self, i, k):
        """
        This method computes the responsability sent from point i to point k.
        
        Parameters
        ----------
        i : integer
                This is the index i needed to select the first point.

        k : integer
                This is the index k needed to select the first point.

        Returns
        -------
        r : float
                This is the responsability of point i for k.
        """

        r = self.__S[i, k] - max([self.__A[i, k_prime] + self.__S[i, k_prime] for k_prime in range(self.__N) if k_prime != k])
        return r

    def __s(self, x_i, x_k):
        """
        This method computes the similarity between two points (negative squared error).
        
        Parameters
        ----------
        x_i : numpy array
                This is the ith point of the dataset.

        x_k : numpy array
                This is the kth point of the dataset.

        Returns
        -------
        s : float
                This is the similarity between points i and k.
        """

        s = -np.sum((x_i - x_k) ** 2)
        return s

    def __GenerateSimilarityMatrix(self):
        """
        This method generates the similarity matrix for all the points given to the mapper.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # As the matrix is symetric, we don't compute the similarity for each couple
        for r in range(self.__N): # For each row
            for c in range(r + 1, self.__N): # For each column
                tmp = self.__s(self.__data[r], self.__data[c])
                self.__S[r, c] = tmp
                self.__S[c, r] = tmp

        # For diagonal: compute "preferences"
        flatS = self.__S.flatten()
        for i in range(self.__N):
            self.__S[i, i] = np.median(flatS[flatS != 0.0])
                            
    def ExecuteAffinityPropagation(self, iterations, lambdaValue = 0.5):
        """
        This method executes the Affinity Propagation algorithm on several iterations.
        
        Parameters
        ----------
        iterations : positive integer
                This is the number of iterations the algorithm will be executed.

        lambdaValue : float
                This is the lambda specified in the paper.

        Returns
        -------
        self.__centers : list
                This list contains all the centroids computed by the algorithm.
        """
        
        # Compute the similarity matrix
        self.__GenerateSimilarityMatrix()

        for i in range(iterations):
            # Update r(i, k) given a(i, k)
            for i in range(self.__N): # For each row
                for k in range(self.__N): # For each column
                    self.__R[i, k] = (1 - lambdaValue) * self.__r(i, k) + lambdaValue * self.__R[i, k]

            # Update a(i, k) given r(i, k)
            for i in range(self.__N): # For each row
                for k in range(self.__N): # For each column
                    self.__A[i, k] = (1 - lambdaValue) * self.__a(i, k) + lambdaValue * self.__A[i, k]

            # Combine both a(i, k) and r(i, k) to get centers
            self.__centers = [i for i in range(self.__N) if self.__R[i, i] + self.__A[i, i] > 0]
            
        return self.__centers	
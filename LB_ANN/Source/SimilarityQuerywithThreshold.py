#Build the Annoy tree and find the nearest neighbors
def Sequence_preprocessing(windowSize, Querysequence):
    w = windowSize
    annoy_indices = []
    seq_length = Querysequence.shape[0]  # Get the sequence length dynamically
    feature_dim = Querysequence.shape[1]  # Get the feature dimensions
    for i in range(seq_length):
        start_idx = max(0, i - w)
        end_idx = min(seq_length, i + w)
        Partition_sequencev = Querysequence[start_idx:end_idx, :]#Create an Annoy Index
        t = AnnoyIndex(feature_dim, 'euclidean')  # Use Euclidean distance
        for idx, vector in enumerate(Partition_sequencev):
            t.add_item(idx, vector)
        t.build(10)  # Build trees; the number of trees can be adjusted as needed.
        annoy_indices.append(t)
        del t  #Free up memory
    return annoy_indices
 #Calculate the min_values_per_column
def Get_min_values_per_column(annoy_indices, candidate_sequence):
    seq_length = candidate_sequence.shape[0]  # Get the sequence length dynamically
    min_values_per_column = []
    for i in range(seq_length):
        candidatepoint = candidate_sequence[i, :]
        annoy_index = annoy_indices[i] #Query the vector’s index for nearest neighbor search
        nearest_index = annoy_index.get_nns_by_vector(candidatepoint, 1,2)[0]
        distances = annoy_index.get_nns_by_vector(Querysequence[nearest_index], n=1, include_distances=True)[1][0]
        min_values_per_column.append(distances)
    return min_values_per_column

#Customizable distance (can be modified)
def distance(p1, p2):
    #X=np.linalg.norm(p1 -p2)
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)

#The original constraint DTW algorithm,s1 is the query sequence,and s2 is the candidate sequence
def DTW(s1, s2, windowSize):
    DTW = {}
    w = max(windowSize, abs(len(s1) - len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i + w)] = float('inf')
        DTW[(i, i - w - 1)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return DTW[len(s1) - 1, len(s2) - 1]

# Attempt to compute
def TrialComputation(s1, s2, Winit, bestdist, columnH):
    # Query sequence Q, :s1
    # candidate sequence C, :s2
    # DTW distance threshold ρ:threshold
    DTW = {}
    w = Winit
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i + w, i)] = float('inf')
        DTW[(i - w - 1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        d = float('inf')
        for j in range(max(i - w, 0), min(i + w, len(s2))):
            dist = distance(s1[j], s2[i])
            #dist = np.linalg.norm(s1[i] - s2[j])
            DTW[(j, i)] = dist + min(DTW[(j - 1, i)], DTW[(j, i - 1)], DTW[(j - 1, i - 1)])
            if (d > DTW[(j, i)]):
                d = DTW[(j, i)]
        if d + columnH[i] > bestdist:
            return -1,-1
    earlyabandcolumn=i #assignment
    if DTW[len(s1) - 1, len(s2) - 1] > bestdist:
        return -1, earlyabandcolumn #Return earlyabandcolumn
    else:
        return DTW[len(s1) - 1, len(s2) - 1] , 0


# DTW with early abandoning
def DTW_earlyabandon(s1, s2, windowSize, bestdist, columnH,earlyabandcolumn):
    #ea：After running the code, Attempt to compute  the returned value.
    DTW = {}
    w = windowSize
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i + w, i)] = float('inf')
        DTW[(i - w - 1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        d = float('inf')
        for j in range(max(i - w, 0), min(i + w, len(s2))):
            dist = distance(s1[j], s2[i])
            DTW[(j, i)] = dist + min(DTW[(j - 1, i)], DTW[(j, i - 1)], DTW[(j - 1, i - 1)])
            if (d > DTW[(j, i)]):
                d = DTW[(j, i)]
        if i>=earlyabandcolumn :
          if d + columnH[i] > bestdist:
            return -1
    if DTW[len(s1) - 1, len(s2) - 1] > bestdist:
        return -1
    else:
        return DTW[len(s1) - 1, len(s2) - 1]

#Normalization Method 1
def normalize(aserie):
    # data structure: DataFrame [ dim1 [...], dim2 [...], ...] ]
    nmSerie = []
    for d in range(aserie.shape[0]):
        oneDim = list(aserie[d])
        mi = min(oneDim)
        ma = max(oneDim)
        dif = (ma - mi) if (ma-mi)>0 else 0.0000000001
        nmValue = [(x-mi)/dif for x in oneDim]
        nmSerie.append(nmValue)
    return np.array(nmSerie)


import numpy as np
import math
import time

from annoy import AnnoyIndex

if __name__ == '__main__':
    ##########Initialize Parameters
    threshold = input("Please input the threshold：")
    sorted_resultsNP = []
    threshold = float(threshold)
    n_trees=10
    dtwtime1=0
    dtwtime2=0
    windowSize = 20
    ###########


    ##############Load Data and Initialize Variables
    for i in range(0, 1): #Data - ERing, Length - 65, Quantity - 300
        #The first data point in the top 30%: 1/(300*0.3)
        cost_matrices = []
        column = []
        columnHHH = []
        sorted_resultsN = []
        candidate_sequences = []
        results = []
        file_name = f'C:/Users/25674/Desktop/LB_Ann/SimilarityQuerwithThreshold/wenjian/{i}_data.npy'
        Querysequence = normalize(np.load(file_name, allow_pickle=True))
        Partition_sequenceves = Sequence_preprocessing(windowSize, Querysequence)
        # Traverse the files of the last 210 arrays
        for j in range(90, 300):
            #The data in the bottom 70 %：300*0.7
            file_name = f'C:/Users/25674/Desktop/LB_Ann/SimilarityQuerwithThreshold/wenjian/{j}_data.npy'
            candidate_sequence = normalize(np.load(file_name))
            candidate_sequences.append(candidate_sequence)
            min_values_per_column = Get_min_values_per_column(Partition_sequenceves, candidate_sequence)
            sum_of_min_per_column = np.sum(min_values_per_column)
            columnHHH.append(min_values_per_column)
            column.append(sum_of_min_per_column)
    #########

        ##############Research on Pre-sorting and Pre-exclusion in Data Filtering
        LBs = column
        LBSortedIndex = np.argsort(LBs)
        ##############

        #############Perform a Given Threshold Query
        start_time = time.time()
        num=0
        num1 = 0
        num2 = 0
        num3 = 0
        for a in range(0, len(LBSortedIndex)):
            b = LBSortedIndex[a]
            HALL = column[b]
            if HALL >  threshold:
                num1 = num1 + 1
                break
            else:
                s1 = Querysequence
                s2 = candidate_sequences[b]
                columnHzb = columnHHH[b]
                columnH = np.cumsum(columnHzb[::-1])[::-1]
                gengxin,earlyabandcolumn = TrialComputation(s1, s2, 5, threshold, columnH)  # our meth
                num=num+1
                if gengxin != -1:
                   num2=num2+1
                   sorted_resultsN.append(gengxin)
                else:
                   earlyabandcolumn=earlyabandcolumn
                   gengxin1 = DTW_earlyabandon(s1, s2, 20, threshold, columnH,earlyabandcolumn)
                   threshold = threshold
                   if gengxin1 != -1:
                       sorted_resultsN.append(gengxin1)
                       num3 = num3 + 1
        end_time = time.time()
        dtwtime = end_time - start_time
        #################


        ################OutPut the result:Performance of the trial-computation strategy for similarity search under varying thresholds.
        print("Number of Sequences Undergoing Trial Computation")
        print(num)
        print("Number of Pre-Excluded Sequences")
        print(num1)
        print("Number of Sequences Directly Identified by Trial Computation")
        print(num2)
        result = sorted([x for x in sorted_resultsN if x < threshold])
        print(result)
        print("Actual Number of Sequences Satisfying the Threshold")
        print(len(result))
        print("Total Processing Time")
        print(dtwtime)
        ################









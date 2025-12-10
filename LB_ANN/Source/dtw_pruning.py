def Sequence_preprocessing(windowSize, Querysequence):#Split the query sequence into n Annoy trees containing 2w nodes.
    w = windowSize
    annoy_indices = []
    seq_length = Querysequence.shape[0]
    for i in range(seq_length):
        start_idx = max(0, i - w)
        end_idx = min(seq_length, i + w)
        Partition_sequencev = Querysequence[start_idx:end_idx, :]
        annoy_indices.append(Partition_sequencev)
    return annoy_indices

def Get_min_values_per_column(annoy_indices, candidate_sequence,threshoid):
    min_values_per_column = []
    Cumulative_Value = 0
    for a in range(0, len(candidate_sequence)):
        candidatepoint = candidate_sequence[a, :]
        annoy_index = annoy_indices[a]
        distances = np.sqrt(np.sum((annoy_index - candidatepoint) ** 2, axis=1))
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        Cumulative_Value = Cumulative_Value + min_distance
        if Cumulative_Value > threshoid:
            return -1,-1
        else:
            min_values_per_column.append(min_distance)
    return min_values_per_column,Cumulative_Value

def Get_min_values_per_column(Querysequence,annoy_indices, candidate_sequence,threshoid):
    LB_Ann = []
    Cumulative_Value = 0
    for i in range(len(candidate_sequence) - 1, -1, -1):
        candidatepoint = candidate_sequence[i, :]
        annoy_index = annoy_indices[i]
        if i == len(candidate_sequence) - 1:
            end_element = Querysequence[i]
            min_distance = np.sqrt(np.sum((candidatepoint - end_element) ** 2))
        elif i == 0:
            first_element = candidate_sequence[i, :]
            min_distance =  np.sqrt(np.sum((candidatepoint - end_element) ** 2))

        else:
            distances = np.sqrt(np.sum((annoy_index - candidatepoint) ** 2, axis=1))
            min_distance = np.min(distances)
        Cumulative_Value = Cumulative_Value + min_distance
        if Cumulative_Value > threshoid:
           return -1,-1
        else:
           LB_Ann.append(Cumulative_Value)
    return LB_Ann,Cumulative_Value

def distance(p1, p2):
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)

def DTW(s1, s2, windowSize):#The original constraint DTW algorithm,s1 is the query sequence,and s2 is the candidate sequence
    DTW = {}
    w = max(windowSize, abs(len(s1)-len(s2)))
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0,i-w),min(len(s2),i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return DTW[len(s1)-1, len(s2)-1]

def DTW_earlyabandon(s1, s2, windowSize, bestdist,columnH):
    # DTW with early abandoning
    DTW = {}
    w = windowSize
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i+w, i)] = float('inf')
        DTW[(i-w-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        d=float('inf')
        for j in range(max(i-w,0),min(i+w,len(s2))):
            dist = distance(s1[j], s2[i])
            DTW[(j, i)] = dist + min(DTW[(j-1, i)],DTW[(j, i-1)], DTW[(j-1, i-1)])
            if (d>DTW[(j,i)]):
                d = DTW[(j,i)]
        if d + columnH[i]>=bestdist:
            return -1
    if DTW[len(s1)-1, len(s2)-1] > bestdist:
        return bestdist
    else:
     return DTW[len(s1)-1, len(s2)-1]

def distance(p1, p2):
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)

def dtw_pruning(s1, s2, windowSize,bestdist,columnH):
    #Since constrainting the window size is equivalent to pruning, this pruning operation
    # is only used for accelerated DTW computation when the window size is not set.
    n, m = len(s1), len(s2)
    w = windowSize
    UB = bestdist
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    sc = 1
    ec = 1
    for i in range(1, n + 1):
        ec_prev = ec
        ec = 1
        smaller_found = False
        sc_next = sc
        start_j = max(sc, i - w)
        end_j = min(m, i + w)
        if start_j ==  end_j:
            return -1
        for j in range(start_j, end_j + 1):
            if j > ec_prev:
                break
            # 计算局部成本
            dist = distance(s1[j-1], s2[i-1])
            # 动态规划递推
            D[j, i] = dist + min(D[j - 1,i], D[j, i - 1], D[j - 1, i - 1])
            if D[j, i] + columnH[i-1] > UB:
                if not smaller_found:
                    sc_next = j + 1
                if j >= ec_prev:
                    break
            else:
                smaller_found = True
                ec = j + 1
        sc = sc_next
    return D[len(s1), len(s2)]


def normalize(arr):#Normalization
    # Calculate the min and max for each column to perform feature scaling.
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)

    # Avoid division by zero error
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1e-10, ranges)

    # Min-Max Normalization
    normalized = (arr - min_vals) / ranges

    return normalized


# ========================
# CONFIGURATION AND SETUP
# ========================

# ========================
# MAIN PROGRAM EXECUTION
# ========================
# Below is the main program logic


time0 = 0#The time taken to construct an Annoy tree for the query sequence
time1 = 0#Time consumption for pre sorting candidate sequences
time2 = 0# Time consumption of the main program
dtwtime = 0#Computation Time of the original DTW

num0 = 0
num1 = 0#Number of sequences pre excluded based on the lower bound function
num2 = 0#The number of sequences with early termination

h = 0

import numpy as np
import math
import time
from scipy.spatial import cKDTree
from annoy import AnnoyIndex

#Find the top N most similar sequences
K = input("Please input the K value to find the top K most similar sequences:")
K = int(K)
#Data Loading and Preprocessing
# The ERing dataset consists of 300 sequences, with the top 30% selected and 90 sequences used as query sequences. An Annoy tree is constructed for each sequence.
######################

for i in range(0,172):
    column = []
    LB_AnnHHH = []
    candidate_sequences = []
    candidate_sequences1 = []
    results = []
    Pre_sorting_one =[]
    windowSize = 300#Configurable as needed
    file_name = f'C:/Users/25674/Desktop/LB_Ann/Data/ArticularyWordRecognition/{i}_data.npy'
    data = np.load(file_name, allow_pickle=True).astype(np.float64)
    Querysequence = normalize(data)
    start_time = time.time()
    annoy_indices = Sequence_preprocessing(windowSize, Querysequence)
    end_time = time.time()
    DTWtime = end_time - start_time
    time0 = time0 + DTWtime
######################

# Take the last 70% sequences of the ERing dataset, totaling 210 sequences, as candidate sequences, and pre sort these sequences.
######################

    for j in range(172, 575):#Configurable as needed
        h=h+1
        file_name = f'C:/Users/25674/Desktop/LB_Ann/Data/ArticularyWordRecognition/{j}_data.npy'
        data = np.load(file_name, allow_pickle=True).astype(np.float64)
        candidate_sequence = normalize(data)
        candidate_sequences.append(candidate_sequence)
        start_time = time.time()
        point_distances = np.sqrt(np.sum((candidate_sequence - Querysequence) ** 2, axis=1))
        accumulated_distance = np.sum(point_distances)
        Pre_sorting_one.append(accumulated_distance)
        end_time = time.time()
        DTWtime = end_time - start_time
        time1 = time1 + DTWtime
######################


#In the process of accumulating and calculating the lower bound value for each candidate sequence, if it is found to exceed the threshold, the sequence will be excluded in advance.
######################
    start_time = time.time()
    Pre_sorting_oneIndex = np.argsort(Pre_sorting_one)
    a = Pre_sorting_oneIndex[0]
    s1 = Querysequence
    s2 = candidate_sequences[a]
    threshold = DTW(s1, s2, windowSize)
    for a in range(0, len(Pre_sorting_oneIndex)):
        b=Pre_sorting_oneIndex[a]
        candidate_sequence1 = candidate_sequences[b]
        LB_Ann,sum_of_min_per_column = Get_min_values_per_column(Querysequence,annoy_indices, candidate_sequence1,threshold)
        if LB_Ann == -1:
            num0 = num0 + 1
            continue
        else:
            candidate_sequences1.append(candidate_sequence1)
            LB_AnnHHH.append(LB_Ann)
            column.append(sum_of_min_per_column)
######################

#Sort the candidate sequences based on the calculated LB_Ann lower bound value, and pre exclude candidate sequences that do not meet the query criteria based on a threshold. Then, use early termination and pruning to calculate the DTW distance for the remaining sequences and search for the top K sequences.
######################
    LBs = column
    LBSortedIndex = np.argsort(LBs)
    if len(LBSortedIndex) == 0:
        continue
    else:
     for a in range(0, len(LBSortedIndex)):
        b = LBSortedIndex[a]
        HALL = column[b]
        s1 = Querysequence
        s2 = candidate_sequences1[b]
        LB_Annzb = LB_AnnHHH[b]
        LB_Ann0 = LB_Annzb[::-1]
        LB_Ann =  np.concatenate([LB_Ann0[1:], [0]])
        Updatedtw =dtw_pruning(s1,s2, windowSize, threshold, LB_Ann0)
        if Updatedtw != -1 and Updatedtw<10000:
            threshold = Updatedtw
        else:
            threshold = threshold
            num2 = num2 + 1
    end_time = time.time()
    DTWtime = end_time - start_time
    time2 = time2 + DTWtime


for i in range(0,1):
    file_name = f'C:/Users/25674/Desktop/LB_Ann/Data/ERing/{i}_data.npy'
    data = np.load(file_name, allow_pickle=True).astype(np.float64)
    s1 = normalize(data)
    for j in range(1, 10):
        file_name = f'C:/Users/25674/Desktop/LB_Ann/Data/ERing/{j}_data.npy'
        data = np.load(file_name, allow_pickle=True).astype(np.float64)
        s2 = normalize(data)
        start_time = time.time()
        dtw_distances = DTW(s1, s2, windowSize)
        end_time = time.time()
        DTWtime = end_time - start_time
        dtwtime = dtwtime + DTWtime
originaldtwtime = dtwtime/9*h




print('Lower Bound Pruning Count',num0)
print('Sequences Excluded by LB',num1)
print('Pruning Count and Early Termination Count',num2)
print('Total Pruned Count ',num0+num1+num2)


print('Measure query sequence processing time',time0)
print('Preprocessing time for the sequences to be matched',time1)
print('Main Program Time',time2)
print('Total Processing Time',time1+time2+time0)
print(originaldtwtime/(time1+time2+time0))
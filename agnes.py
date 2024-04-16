#!/usr/bin/env python

import os.path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core import frame
from pandas.core.interchange import dataframe
from tabulate import tabulate


####################
# DRAW CLUSTERS    #
####################

# Plot the data points in Euclidean space, color-code by cluster
def DrawClusters(dataframe):
    sns.relplot(data=dataframe, x=dataframe.columns[0], y=dataframe.columns[1], hue='clusters', aspect=1.61, palette="tab10")
    plt.show()
        
####################
# LOAD DATA        #
####################
def LoadData(DB_PATH):

    # Load the input file into a Pandas DataFrame object
    dataframe = pandas.read_csv(DATA_PATH, sep=';', encoding='cp1252')

    # Check how many rows and columns are in the loaded data
    assert dataframe.shape[1] == 22, "Unexpected input data shape."

    # TODO: Perform a PROJECT operation to filter down to the following attributes:
    #       - latitude
    #       - longitude
    dataframe = dataframe[['latitude','longitude']]
    dataframe.dropna(inplace=True)
    print(dataframe)
    assert dataframe.shape[1] == 2, "Unexpected projected data shape."
    return dataframe

####################
# GET NUM CLUSTERS #
####################
def GetNumClusters(dataframe):
    
    # TODO: Get the number of unique clusters
    num_clusters = dataframe["clusters"].nunique()
    print(num_clusters)
    return num_clusters

####################
# GET CLUSTER IDS  #
####################
def GetClusterIds(dataframe):
    
    # TODO: Get the unique IDs of each cluster
    cluster_ids = dataframe['clusters'].unique()
    print(cluster_ids)
    return cluster_ids

####################
# GET CLUSTER      #
####################
def GetCluster(dataframe, cluster_id):
    
    # TODO: Perform a SELECT operation to return only rows in the specified cluster
    cluster = dataframe[dataframe['clusters'] == cluster_id]
    print(cluster)
    return cluster

####################
# DISTANCE         #
####################
def Distance(lhs, rhs):

    # TODO: Calculate the Euclidean distance between two rows
    dist = np.sqrt((lhs['latitude'] - rhs['latitude'])**2 + (lhs['longitude'] - rhs['longitude'])**2)
    
    return dist

####################
# SINGLE LINK DIST #
####################
def SingleLinkDistance(lhs, rhs):

    # TODO: Calculate the single-link distance between two clusters
    # Initialize mindist to inf so it initially always passes
    mindist = float('inf')
    for index_lhs in lhs.index:
        for index_rhs in rhs.index:
            dist = Distance(lhs.loc[index_lhs], rhs.loc[index_rhs])
            if dist < mindist:
                mindist = dist
    return mindist

######################
# COMPLETE LINK DIST #
######################
def CompleteLinkDistance(lhs, rhs):

    # TODO: Calculate the complete-link distance between two clusters
    maxdist = 0
    for lhs_index in lhs.index:
        for rhs_index in rhs.index:
            dist = Distance(lhs.loc[lhs_index], rhs.loc[rhs_index])
            if dist > maxdist:
                maxdist = dist
    return maxdist

#######################
# RECURSIVELY CLUSTER #
#######################
def RecursivelyCluster(dataframe, K, M):

    # TODO: Check if we have reached the desired number of clusters
    currentClusters = GetClusterIds(dataframe)
    if len(currentClusters) <= K:
        return dataframe
    # TODO: Find the closest 2 clusters
    cluster_to_merge = (None, None)
    minDist = float('inf')
    for i in range(len(currentClusters)):
        for j in range(i+1,len(currentClusters)):
            cluster_i = dataframe[dataframe['clusters'] == currentClusters[i]]
            cluster_j = dataframe[dataframe['clusters'] == currentClusters[j]]
            dist = M(cluster_i, cluster_j)
            if dist < minDist:
                minDist = dist
                cluster_to_merge = (currentClusters[i], currentClusters[j])
    print(dist)
    # TODO: Merge the closest 2 clusters
    dataframe.loc[dataframe['clusters'] == cluster_to_merge[1], 'clusters'] = cluster_to_merge[0]

    # Recurse with the updated DataFrame
    return RecursivelyCluster(dataframe, K, M)

####################
# AGNES            #
####################
def Agnes(db_path, K, M):

    # Load the data in and select the features/attributes to work with (lat, lon)
    dataframe = LoadData(DATA_PATH)
    assert dataframe.shape[1] == 2, "Unexpected input data shape (lat, lon)."

    # TODO: Add each datum to its own cluster (as a new column)
    dataframe['clusters'] = dataframe.index
    assert dataframe.shape[1] == 3, "Unexpected input data shape (lat, lon, cluster)."
    
    # Generate clusters from all points and recursively merge
    results = RecursivelyCluster(dataframe, K, M)
    # Graphs the clusters
    DrawClusters(results)
    return results
    
####################
# MAIN             #
####################
if __name__=="__main__":

    RUN_UNIT_TEST=False
    if RUN_UNIT_TEST:
        # Path where you downloaded the data
        DATA_PATH = './unit_test_data.csv'
        df = LoadData(DATA_PATH)
        K=2 # The number of output clusters.
        M=SingleLinkDistance # The cluster similarity measure M to be used.
    
        # Run the AGNES algorithm with the unit test data
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (5,3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_unit_test.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
    
    # TODO: When you are ready to run with the full dataset, modify the following line to True
    RUN_FULL_SINGLE_LINK=False
    if RUN_FULL_SINGLE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K=6 # The number of output clusters.
        M=SingleLinkDistance # The cluster similarity measure M to be used.
        
        # Run the AGNES algorithm using single-link
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (100,3), "Unexpected output data shape. {}".format(results.shape)
        
        # Write results to file
        f = open("agnes_single_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

    # TODO: When you are ready to run with the full dataset, modify the following line to True
    RUN_FULL_COMPLETE_LINK=True
    if RUN_FULL_COMPLETE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K=6 # The number of output clusters.
        M=CompleteLinkDistance # The cluster similarity measure M to be used.
        
        # Run the AGNES algorithm using complete-link
        results = Agnes(DATA_PATH, K, M)
        #assert results.shape == (100,3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_complete_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

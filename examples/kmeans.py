'''
Created on 23 dic. 2018

@author: Johan Mejia
'''

## Libraries ###########################################################
import sys, os, tqdm, cv2
sys.path.append('./libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from IoU import intersection_over_union

## Path and global variables ###########################################
objIoU = intersection_over_union() # Object for a IoU estimate

## Functions and Class #################################################
def database_gen(directories):
    if type(directories) != list: directories = [directories]
    data = np.array([]).reshape((0,2))    

    for imdir in directories:
        print("[INFO] Reading images from " + imdir)
        for ifile in tqdm.tqdm(os.listdir(imdir)):
            ifile = os.path.join(imdir, ifile)
            if os.path.isfile(ifile): # Read height, width for each image
                h, w = cv2.imread(ifile).shape[:2]
                data = np.append(data, [[w,h]], axis = 0)
            else:
                print("[WARNING] {} invalid. Ignored it.".format(ifile))
    
    return data

def ioU_clusters(data, total_clusters = 10, isplot = False):
    if not hasattr(total_clusters, '__iter__'): total_clusters = range(1, total_clusters + 1)
    N = len(total_clusters)

    rowsNum = int(np.floor(np.sqrt(N)))
    columnsNum = int(np.ceil(N/float(rowsNum)))
    clustersCenter = [] # Save a clusters centers
    IoU = np.zeros(N) # Save the result of the mean of maximum IoU between the bounding box and individual anchors.
    if isplot: plt.figure(1, figsize = (10,4))
    
    print("[INFO] KMeans process...")
    for i, clus in tqdm.tqdm(enumerate(total_clusters)):
        # Estimate of clusters
        kmeans = KMeans(n_clusters = clus, max_iter = 22).fit(np.log(data)) # Define clusters with log-function
        c_pred = kmeans.predict(np.log(data))
        
        # Plot
        if isplot:
            plt.subplot(rowsNum, columnsNum, i + 1)
            plt.scatter(np.log(data[:,0]), np.log(data[:,1]), c = c_pred, s = 0.01)
            plt.xlabel("Width"); plt.ylabel("Height")
        
        # Save each cluster centers
        kmeans = np.exp(kmeans.cluster_centers_) # Clusters centers = [width, height]
        clustersCenter.append(kmeans)
        
        # Estimate of IoU between clusters and bounding boxes
        IoUmatrix = np.zeros((len(data), clus)) # Matrix estimate IoU
        aux = np.zeros(len(data))
        for pos in range(len(data)):
            for numClus in range(len(kmeans[:,0])):
                IoUmatrix[pos,numClus] = objIoU.IoUestimate(data[pos,:], kmeans[numClus,:])
            aux[pos] = max(IoUmatrix[pos,:]) # Save max(IoU(Xdata,clustersCenter[n]))
        IoU[i] = np.mean(aux) # Save max(IoU(Xdata,clustersCenter))
    
    return IoU, clustersCenter

## Main ################################################################
def kmeans_process(total_clusters = 8):
    data = database_gen(['./data/Train', './data/Test'])
    meanIoU, total_clusters = ioU_clusters(data, total_clusters = total_clusters, isplot = True)
    print()
    clusters = {}
    for i, center in enumerate(total_clusters):
        clusters[len(center)] = center
        print('[INFO]: NC#{} -> mean(max(IoU)) = {}'.format(len(center), meanIoU[i]))
        # columnName.append('Cluster'+str(n+2))

    print("[INFO]: Cluster's center: ", clusters)

    plt.figure(2, figsize = (10,4))
    plt.plot(list(clusters.keys()), meanIoU, '-o', markersize = 10)
    plt.xlabel("# Clusters"), plt.ylabel("mean(max(IoU))")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    kmeans_process(total_clusters = 10)

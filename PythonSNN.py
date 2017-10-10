#Author: Amar Mani Aryal.
<<<<<<< HEAD
#Implementation of SNN on Python 
=======
#Implementation of SNN on Python by Ertöz et al. (2003).
>>>>>>> f921cf8720e35cc3daf3eb35daca2ef0dfcd4577
#Requires gmaps for google map visualization
import sys
import csv, io
import os
import time
import pandas as pd
import numpy as np
import gmplot
import webbrowser
from gmplot import *
import gmaps #gmaps for google map visualization
#Main
if __name__=="__main__":
    #k,minPoints,and eps are user defined inputs
    k=28#Nearest Neighbor Size
    minPoints=13#defines core point
    eps=13 #defines noise
fileName='June.csv'# Specify Input file

#Method to read the File Containing the Locations and Time
def readTextFile():
	#Give appropriate input file path
    fileContent = open("/home/student/Input/June.csv", "r")
    #if header Present:
    initialIndex=1
    #else:
    #initialIndex=0
    lines = fileContent.readlines()
    latitude=[]
    longitude=[]
	#Provide a different logic for different input file formats. 
	#This Implementation has input files having latitude and longitude seperated by a ','
    for i in range(initialIndex,len(lines)):
        [lat,lon]= str(lines[i]).split(",")
        if(float(lat)<>0 and float(lon)<>0):
            latitude.append(lat)
            longitude.append(lon)
    return [latitude,longitude]

#Method to compute Distance
# Define other distance functions here
def eucledian_dist(lat1,lon1,lat2, lon2):
    return (abs(float(lat2)-float(lat1))**2+abs(float(lon2)-float(lon1))**2)**0.5

#Method to compute paiwise distance 
def computeDistanceMatrix(lat,longi):
    distanceMatrix=[[0 for i in range(len(lat))] for j in range(len(lat))]
    for i in range(len(lat)):
        for j in range(len(longi)):
            distanceMatrix[i][j]=((i,j),eucledian_dist(lat[i],longi[i],lat[j],longi[j]))
    return distanceMatrix

#Method to compute K-Nearest Neighbors of points
def findKNNList(distanceMatrix, k):
    count=len(distanceMatrix)
    # matrix = [[0 for i in range(count)] for j in range(count)]#matrix with dimension count*count
    global similarityMatrix# list to hold k similar indices of points to any point
    count=len(distanceMatrix)
    similarityMatrix = [[0 for i in range(k)] for j in range(count)]
    for i in range(count):
        matrixsorted = sorted(distanceMatrix[i], key=lambda x: x[1])# sort each row of Matrix based on distance(value)
        #print matrixsorted
        for j in range(k):
            #file.write(str(matrixsorted[0][j+1]) + " ")# write all the k nearest neighbors to file
            similarityMatrix[i][j] = int(matrixsorted[j+1][0][1])#Assign each row  in similarityMatrix to all the k nearest neighbors..
    return similarityMatrix
#---------end of findKNNList---------------------------------
   
def f(k1,k2,i):
    if i==k1:
        return k2
    else:
        return k1

# Step 1: Preprocess dta and find K nearest neighbors-----------------------------------------
[lat,lon]= readTextFile()
x= zip(lat,lon,lat)
distanceMatrix=computeDistanceMatrix(lat,lon)#Compute Upper Traingular PairWise Distance Matrix
similarityMatrix=findKNNList(distanceMatrix,k)
count=len(lat)

#Step 2: Construct SNN graph from the sparsified matrix---------------------------------------
def countIntersection(listi,listj):
    intersection=0
    for i in listi:
        if i in listj:
            intersection=intersection+1
    return intersection

def sharedNearest(count,k):
    Snngraph= [[0 for i in range(count)] for j in range(count)]
    for i in range(0,count-1):
        nextIndex=i+1
        for j in range(nextIndex,count):
            if j in similarityMatrix[i] and i in similarityMatrix[j]:
                count1=countIntersection(similarityMatrix[i],similarityMatrix[j])
                Snngraph[i][j]=count1
                Snngraph[j][i]=count1
    return Snngraph
sharedNearestN= sharedNearest(count,k)# count of shared neighbors between points


#Step3:Find the SNN density of each point--------------------------------------------------
def density(x,eps):
    numbPoints=0
    for i in range(0,count):
        if x[i] >= eps:
            numbPoints=numbPoints+1
    return numbPoints
snnDensity1=[None for i in range(len(sharedNearestN))]
for i in range(len(sharedNearestN)):
        snnDensity1[i]=density(sharedNearestN[i],eps)


#Step4: Find the core points-------------------------------------------------------------
def coreornot(x,minPoints):
    if x >= minPoints:
        return True
    else:
        return False
def core(x,y):
    if x >= minPoints:
        return y
    else:
        return None
		
coreOrNot=[None for i in range(len(snnDensity1))]
for i in range(len(snnDensity1)):
    coreOrNot[i]=coreornot(snnDensity1[i],minPoints)
corePointsList1=[]
snnDensity2=zip(snnDensity1, [i for i in range(len(snnDensity1))])
for i in range(len(snnDensity2)):
    corePointsList1.append(core(snnDensity2[i][0],snnDensity2[i][1]))
corePointsList=[ x for x in corePointsList1 if x!=None]#list of core points



#Step5: Find clusters from the core points------------------------------------------------
#If two core points are within Eps radius they belong to the same cluster
def findCoreNeighbors(p,corePts,sharednearestNeighbors,eps):
    coreNeighbors=[]
    p2=None
	
    for i in range(0,len(corePts)):
        p2=corePts[i]
        #if two core points share more than eps neighbors make the core point core nearest neighbor of other
        if(p!=p2 and sharednearestNeighbors[p][p2]>=eps):
            coreNeighbors.append(p2)
    return coreNeighbors

def expandCluster(labels,neighborCore,corePts,C,sharednearestNeighbors,eps,visited):
    while len(neighborCore)>0:
            p=neighborCore.pop(0)
            if p in visited:
                continue
            labels[p]=C
            visited.append(p)
            neighCore=findCoreNeighbors(p,corePts,sharednearestNeighbors,eps)
            neighborCore.extend(neighCore)
    return labels
	
visited=[]#list to store points visited
labels=[0 for i in range(count)]
neighborCore=[]#neighborss of core points
c=0 

for i in range(0,len(corePointsList)):
    p=corePointsList[i]
    if p in visited:
        continue
    visited.append(p)
    c=c+1
    labels[p]=c
    neighborCore = findCoreNeighbors(p, corePointsList, sharedNearestN, eps)
    labels=  expandCluster(labels, neighborCore, corePointsList, c,sharedNearestN, eps, visited)


#Step 5: Compute the final cluster labels------------------------------------------------------------
#All points that are not within a radius of Eps of a core point are discarded (noise)
#Assign all non-noise, non-core points to their nearest core point
for i in range(count):
    notNoise=False
    maxSim=sys.maxint
    bestCore=-1
    sim=None
    if(coreOrNot[i]):#core Point
        continue
    for j in range(len(corePointsList)):
        p=corePointsList[j]
        #sharedNearestN contains count of shared neighbors between points
        # sim gives the similarity  between core point and the other point.
        sim=sharedNearestN[i][p]
        # if sim is greater than eps--> the point is not a noise
        if(sim>=eps):
            notNoise=True
         # if sim is less than eps--> the point is  a noise point assign cluster index 0 to it
        else:
            labels[i]=0
            break
        #Here we attempt to see to which core point does the non-core point has maximum similarity
        if(sim>maxSim):
            maxSim=sim
            bestCore=p
        #End of inner for loop
    #for each non-core point assign the index of core point with which the point has maximum similarity
    if(notNoise):
        labels[i]=labels[bestCore]         
print labels #Labels consist of the cluster index


#visualize all the results---------------------------------------------------------------------------
for i in range(len(lat)):
    lat[i]=float(lat[i])
    lon[i]=float(lon[i])
    labels[i]=int(labels[i])
	
zipppedLoc=list(zip(lat,lon,labels))
zipppedLocList=[list(i) for i in zipppedLoc]
name1=str(fileName.replace('.csv',''))+'k'+str(k)+'eps'+str(eps)+'minpoints'+str(minPoints)
name=name1+'.txt'
#Give appropriate path here
filePaths='/home/student/Result/'+name
filePaths1='/home/student/Result/'+name1
#Write labels to file
outfile=open(filePaths,'w')
for i in range(len(labels)):
    outfile.write(str(labels[i]))
    outfile.write('\n')
	
numberOfClusters=max(labels)
print numberOfClusters
clusterLat = [0 for x in range(numberOfClusters+1)]
clusterLong= [0 for x in range(numberOfClusters+1)]
cluster=[0 for x in range(numberOfClusters+1)]
longMean=float(sum(lon)) / len(lon)
latMean=float(sum(lat)) / len(lat)
zipppedLoc=list(zip(lat,lon,labels))
zipppedLocList=[list(i) for i in zipppedLoc]

for i in range(numberOfClusters+1):
    clusterLat[i]=[t[0] for t in zipppedLoc if t[2] == i]#clusterLat[i] contails all the latitudes in ith cluster
    clusterLong[i]=[t[1] for t in zipppedLoc if t[2] == i]#clusterLat[i] contails all the latitudes in ith clustert
    cluster[i]=[[t[0],t[1]] for t in zipppedLoc if t[2] == i]

#visualize all the clusters to see intersting clusters
df2=pd.DataFrame(np.array(zipppedLocList),columns = list("OLI"))
df2=df2.loc[df2['I'] <> 0]
gmaps.scatter(df2['L'], df2['O'], colors=df2['I'])


#Plot selected clusters in Google Map
interesting=[2,3,14,16]
colors=['#FF0000','#00FF00','#0000FF','#000000','#FF00FF','#FFF000','#FFFFFF','#FFF0F0','#F0FFF0','#F00FFF','#FFF0FB']
gmap = gmplot.GoogleMapPlotter(longMean,latMean, 11)
j=0
pth=filePaths1+'.html'
nameabs=name1+'.html'
url = 'file://'+pth
for i in interesting:
    gmap.scatter( clusterLong[i],clusterLat[i], colors[j], size=200, marker=False)
    j=j+1
	#Save the visualization in html file
    gmap.draw(pth)
    gmap.draw(nameabs)
	
new = 0
#Open the saved html file in browser
webbrowser.open(url,new=new)

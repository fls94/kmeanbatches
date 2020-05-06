# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:22:05 2020

@author: Tu Anh dep trai
"""
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


#initiating centroids from dataset
def init_centroids(k, parsed_dataset):
  centroids = []
  i = 0
  dataset = parsed_dataset.as_numpy_iterator()
  while i < k:
    sample = next(dataset)['vector']
    centroids.append(sample)
    i += 1
  return np.array(centroids)


#assign to clusters
def assign_clusters(assigned,counter,centroids,batch,k):
  D = cdist(batch, centroids)
  cluster_res = np.argmin(D, axis= 1)
  counter_temp = Counter(cluster_res)
  for i in counter_temp:
    counter[i] += counter_temp[i]
  for  i,x  in enumerate(cluster_res):
    assigned[x] += batch[i]
  return assigned, counter


#Stop condition
def stop_condition(centroids, centroids_new):
  return (set([tuple(a) for a in centroids ]) == 
        set([tuple(a) for a in centroids_new]))
  

def kmean(transformed_dataset, shuffled_dataset, k, len_vec):
  centroids = [init_centroids(k,shuffled_dataset)]
  centroids_new = np.zeros((k,len_vec))
  while True:
    counter = {}
    assigned = {}
    for i in range(k):
      counter[i] = 0
      assigned[i] = np.zeros((len_vec))
    iter_dataset = transformed_dataset.as_numpy_iterator()
    for data in iter_dataset:
      assigned, counter = assign_clusters(assigned,counter, centroids[-1],data["vector"],k)
    centroids_new = []
    for i in range(k):
      if counter[i] == 0:
        continue
      else:
        centroids_new.append(assigned[i] / counter[i])
    if stop_condition(centroids[-1], centroids_new) == True:
      break
    centroids.append(centroids_new)
  print("Final result: ",centroids[-1])
  return centroids[-1]

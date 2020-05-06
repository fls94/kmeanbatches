# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:24:36 2020

@author: Tu Anh dep trai
"""
import shutil
import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist


def cluster_folder(dataset, centroids, cluster_path):
  if os.path.exists(cluster_path):
    shutil.rmtree(cluster_path, True)
  os.mkdir(cluster_path)
  iter_dataset = dataset.as_numpy_iterator()
  index = 0
  while True:
    try:
      data = next(iter_dataset)
      vectors = data["vector"]
      cluster_assigned = np.argmin(cdist(vectors, centroids), axis=1)
      enu = enumerate(cluster_assigned)
      for i in set(cluster_assigned):
        data_save = [(data["vector"][k],data["name"][k]) for k,x in enu if x == i]
        pickle.dump(data_save,open(os.path.join(cluster_path,"cluster"+str(i)+"batch"+str(index)+".txt"),"wb"))
      index += 1
    except Exception:
      print("Done saving clusters!")  
      break
      

def join_clusters(len_vec, k, cluster_path):
  for cluster in range(k):
    dataset = []
    records = [os.path.join(cluster_path,i) for i in os.listdir(cluster_path) if "cluster"+str(cluster) in i]
    for i in records:
      dataset.append(pickle.load(open(i,"rb")))
    dataset = sum(dataset, [])
    pickle.dump(dataset, open("cluster"+str(cluster)+".txt","wb"))
      
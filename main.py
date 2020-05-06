# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:26:23 2020

@author: Tu Anh dep trai
"""
import read_records
import cluster
import split


k = 1000      #number of cluster
len_vec = 512     #vectors length
batch_size = 100      #batch size
cluster_dir = "cluster"     #directories for temporary save clusters
filename = "dataset.tfrecord"


def main():

  dataset = read_records.read_records(len_vec, filename)
  transformed_dataset = dataset.batch(batch_size)
  shuffled_dataset = dataset.shuffle(1000)
  centers = cluster.kmean(transformed_dataset, shuffled_dataset, k, len_vec)
  split.cluster_folder(transformed_dataset, centers, cluster_dir)
  split.join_clusters(len_vec, k, cluster_dir)


main()
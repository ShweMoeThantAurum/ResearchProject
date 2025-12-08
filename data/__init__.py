"""
Data preprocessing package for preparing raw datasets and generating
client-specific partitions for federated learning experiments.

Each dataset script:
 - downloads raw data from S3 (if needed)
 - constructs sliding-window sequences
 - splits into train/valid/test
 - partitions nodes into client-specific subsets
"""

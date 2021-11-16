#!/bin/bash

for percent in 30 40 50 70 90 95 100; do
  wget https://cdn.rcsb.org/resources/sequence/clusters/bc-${percent}.out
done
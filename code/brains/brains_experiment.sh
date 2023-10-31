#!/bin/bash

for dataset in "development"
do
    echo "$dataset"
    for signature in "degree" "neighbors"
    do
    	echo "$signature"
    	for metric in "l1" "dot"
    	do
    		echo "$metric"
    		python3 brains.py ../../ $dataset --alg tgw --signature $signature --metric $metric
    	done
    done
done
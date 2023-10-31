#!/bin/bash

datasets="development abide adhd"

classifiers="kNN SVM_linear"

metrics="l1 l2"

# activity
# for dataset in $datasets; do
# 	for classifier in $classifiers; do
# 		echo "training $classifier on corr for $dataset.."
#     	python3 evaluate/brains_train.py $dataset --classifier $classifier --alg corr
#     	echo "training $classifier on tw for $dataset.."
#     	for metric in $metrics; do
#     		python3 evaluate/brains_train.py $dataset --classifier $classifier --alg tw --metric $metric
#     	done
#     done
# done


# tgw
classifiers="SVM_linear"
signatures="degree interaction"
metrics="l1 l2"
Ks="1 2 3"
for dataset in $datasets; do
	for classifier in $classifiers; do
    	for signature in $signatures; do
    		for k in $Ks; do
    			for metric in $metrics; do
    				echo "training $classifier on tgw for $dataset, $signature$k, and $metric..."
    				python3 evaluate/brains_train.py $dataset --classifier $classifier --alg tgw --signature $signature --k $k --metric $metric
    			done
    		done
    	done
    done
done

classifiers="kNN SVM_linear"

# tgkernel
# tgkernels="SEKS"
# tgkernels="SEKS SEWL"
# Ks="1 2"
# for dataset in $datasets; do
# 	for classifier in $classifiers; do
# 		for tgkernel in $tgkernels; do
# 			for k in $Ks; do
#     			echo "training $classifier for $tgkernel on $dataset for k=$k ..."
#     			python3 evaluate/brains_train.py $dataset --alg $tgkernel --k $k --classifier $classifier
#     		done
#     	done
#     done
# done

# Ks="1 2 3 4 5 6 7 8 9 10"
# Ss="100 1000"
# for dataset in $datasets; do
# 	for classifier in $classifiers; do
# 		for S in $Ss; do
# 			for k in $Ks; do
#     			echo "training $classifier for $tgkernel on $dataset for k=$k ..."
#     			python3 evaluate/brains_train.py $dataset --alg "TMAP" --k $k --S $S --classifier $classifier
#     		done
#     	done
#     done
# done
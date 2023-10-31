#!/bin/bash

datasets="infectious_ct1 tumblr_ct1 highschool_ct1 facebook_ct1 dblp_ct1 infectious_ct2 tumblr_ct2 highschool_ct2 facebook_ct2 dblp_ct2 mit_ct1 mit_ct2"
classifiers="SVM_linear SVM_rbf kNN"

signatures="subtrees walks"
metrics="l1 l2"
Ks="0 1 2 3"
dcosts="0 1 2"

# dtgw
for dataset in $datasets; do
	for classifier in $classifiers; do
    	for signature in $signatures; do
    		for k in $Ks; do
    			for metric in $metrics; do
                    for dcost in $dcosts; do
    				    echo "training $classifier on $dataset for dtgw with $signature, and $metric..."
    				    python3 evaluate/dissemination_train.py $dataset --alg dtgw --signature $signature --k $k --metric $metric --dcost $dcost --classifier $classifier
                    done
    			done
    		done
    	done
	done
done

# # tgkernel
# tgkernels="SEKS SEWL LGKS LGWL"
# Ks="1 2 3"
# for dataset in $datasets; do
# 	for classifier in $classifiers; do
# 		for tgkernel in $tgkernels; do
# 			for k in $Ks; do
#     			echo "training $classifier for $tgkernel on $dataset for k=$k ..."
#     			python3 evaluate/dissemination_train.py $dataset --alg $tgkernel --k $k --classifier $classifier
#     		done
#     	done
#     done
# done

# # tgraphlet
# tgkernels="tkg10 tkg11 tkgw"
# Ks="0 1 2 3"
# for dataset in $datasets; do
# 	for classifier in $classifiers; do
# 		for tgkernel in $tgkernels; do
# 			for k in $Ks; do
#     			echo "training $classifier for $tgkernel on $dataset for k=$k ..."
#     			python3 evaluate/dissemination_train.py $dataset --alg $tgkernel --k $k --classifier $classifier
#     		done
#     	done
#     done
# done


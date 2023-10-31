#!/bin/bash
export PYTHONPATH=.

datasets="development abide adhd"
metrics="l1 l2"

# for dataset in $datasets; do
# 	for metric in $metrics; do
#         echo "running tw on $dataset with metric $metric..."
#         python3 brains/brains.py $dataset --alg tw --metric $metric
#     done
#     echo "running corr on $dataset..."
#     python3 brains/brains.py $dataset --alg corr
# done

# tgw
signatures="degree interaction"
metrics="l1 l2"
Ks="1 2 3"
for dataset in $datasets; do
    for signature in $signatures; do
    	for k in $Ks; do
    		for metric in $metrics; do
    			echo "running tgw on $dataset with $signature$k, $metric..."
    			python3 brains/brains.py $dataset --alg tgw --signature $signature --k $k --metric $metric
    		done
    	done
    done
done

# dtgw
signatures="degree interaction"
metrics="l1 l2"
Ks="1 2 3"
for dataset in $datasets; do
    for signature in $signatures; do
    	for k in $Ks; do
    		for metric in $metrics; do
    			echo "running dtgw on $dataset with $signature$k, $metric..."
    			python3 brains/brains.py $dataset --alg dtgw --signature $signature --k $k --metric $metric
    		done
    	done
    done
done


# tgkernel
# for dataset in $datasets;
# do
#     echo "convert temporal graphs of dataset $dataset to txt format."
#     python3 brains/brains.py $dataset --savetxt 1
# done

# cd ../output/brains
# for dataset in $datasets; do
# 	cd $dataset
#     echo "run tgkernel on dataset $dataset"
#     # SE RW kernel
#     # ./../../../code/tgkernel/release/tgkernel "../../../datasets/brains/$dataset/temporal_graphs/$dataset" 11 1 2
#     # approx kernel
#     for S in 100 1000; do
#     # for S in 10000; do
#     	./../../../code/tgkernel/release/tgkernel "../../../datasets/brains/$dataset/temporal_graphs/$dataset" 13 1 10 $S
#     done
#     cd ..
# done

#tgraphlet kernel
# cd ../../code
# for dataset in $datasets;
# do
#     echo "convert temporal graphs of dataset $dataset to tgraphlet's txt format."
#     python3 brains/brains.py $dataset --savetxt 1 --distinct_labels 0
# done

# for dataset in $datasets;
# do
#   cd $dataset
#     echo "run tgkernel on dataset $dataset"
#     for action in 0 1 2; do
#         ./../../../code/tgraphlet/release/tgraphlet "../../../datasets/brains/$dataset/temporal_graphs/$dataset" $action
#     done
#     cd ..
# done

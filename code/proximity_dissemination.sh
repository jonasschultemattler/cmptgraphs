export PYTHONPATH=.

datasets="infectious_ct1 tumblr_ct1 highschool_ct1 facebook_ct1 dblp_ct1 infectious_ct2 tumblr_ct2 highschool_ct2 facebook_ct2 dblp_ct2 mit_ct1 mit_ct2"

# inits="optimistic_warping sigma* optimistic_matching"
inits="diagonal_warping"
signatures="subtrees walks"
Ks="0 1 2 3"
# windows="1 2 3 5 10 20 50 100"
# windows="0.01 0.02 0.05 0.1 0.2 0.3 0.5 1"
windows="0.2"
metrics="l1 l2"
# iters="2 3 4 5 7 10"
iters="5"
costs="0 1 2"
# numbers="10 20 50"
numbers="100"

#dtgw
for number in $numbers; do
	for dataset in $datasets; do
		for init in $inits; do
    		for signature in $signatures; do
    			for k in $Ks; do
    				for metric in $metrics; do
    					for window in $windows; do
    						for iter in $iters; do
                                for cost in $costs; do
    							     python3 dissemination/compute_distances.py $dataset --signature $signature --k $k --init $init --metric $metric --window $window --iterations $iter --number $number --dcost $cost
                                done
    						done
    					done
    				done
    			done
    		done
    	done
	done
done

#tgkernel
# cd ../output/dissemination
# for dataset in $datasets;
# do
# 	cd $dataset
#     echo "run tgkernel on dataset $dataset"
#     for action in 7 8 10 11; do
#         ./../../../code/tgkernel/release/tgkernel "../../../datasets/dissemination/$dataset/$dataset" $action 1 2
#     done
#     cd ..
# done

# #tgraphlet
# cd ../output/dissemination
# for dataset in $datasets;
# do
#   cd $dataset
#     echo "run tgkernel on dataset $dataset"
#     ./../../../code/tgraphlet/release/tgraphlet
#     for action in 0 1 2; do
#         ./../../../code/tgraphlet/release/tgraphlet "../../../datasets/dissemination/$dataset/$dataset" $action
#     done
#     cd ..
# done


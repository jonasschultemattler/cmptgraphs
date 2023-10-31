#!/bin/bash
mkdir datasets
cd datasets
mkdir dissemination
cd dissemination

datasets="dblp_ct1 dblp_ct2 facebook_ct1 facebook_ct2 highschool_ct1 highschool_ct2 infectious_ct1 infectious_ct2 mit_ct1 mit_ct2 tumblr_ct1 tumblr_ct2"
for dataset in $datasets; do
	wget "https://www.chrsmrrs.com/graphkerneldatasets/$dataset.zip"
	unzip "$dataset.zip"
	rm "$dataset.zip"
done
dtgw.so: dtgw.c
	g++ -c -fpic -Wall -std=c++17 -mavx2 -ftree-vectorize -march=haswell -maes -mno-pku -mno-sgx --param l1-cache-line-size=64 --param l1-cache-size=32 --param l2-cache-size=33792 dtgw.c
	g++ -shared -o build/dtgw.so dtgw.o
	rm -f dtgw.o

# warping_test.so: warping_test.c lap.cpp
# 	g++ -c -fpic -Wall warping_test.c lap.cpp
# 	g++ -shared -o warping_test.so warping_test.o lap.o
# 	rm -f warping_test.o lap.o
# warping_test.so: dtgw.c lap.cpp
# 	g++ -c -fpic -Wall dtgw.c lap.cpp
# 	g++ -shared -o dtgw.so dtgw.o lap.o
# 	rm -f dtgw.o lap.o
# warping_test.so: dtgw.cpp lap.cpp
# 	g++ -c -fpic -Wall dtgw.cpp lap.cpp
# 	g++ -shared -o dtgw.so dtgw.o lap.o
# 	rm -f dtgw.o lap.o
# warping_test.so: dtgw.cpp
# 	g++ -c -fpic -Wall dtgw.cpp
# 	g++ -shared -o dtgw.so dtgw.o
# 	rm -f dtgw.o
# warping_test.so: dtgw.c lapjv.h
# 	g++ -c -fpic -Wall dtgw.c lapjv.h
# 	g++ -shared -o dtgw.so dtgw.o lapjv.o
# 	rm -f dtgw.o lapjv.o
# dtgw.so: dtgw.c
# 	gcc -c -fpic -Wall dtgw.c
# 	gcc -shared -o dtgw.so dtgw.o
# 	rm -f dtgw.o
# dtgw.so: dtgw.c
# 	g++ -c -fpic -Wall -std=c++17 -mavx2 -ftree-vectorize -march=haswell -maes -mno-pku -mno-sgx dtgw.c
# # 	g++ -c -fpic -Wall -std=c++17 -mavx2 dtgw.c
# 	g++ -shared -o build/dtgw.so dtgw.o
# 	rm -f dtgw.o
# dtgw.so: dtgw.c
# 	g++ -c -fpic -Wall -fopenmp -std=c++17 -mavx2 -ftree-vectorize -march=haswell -maes -mno-pku -mno-sgx dtgw2.c
# # 	g++ -c -fpic -Wall -std=c++17 -mavx2 dtgw.c
# 	g++ -shared -o build/dtgw.so dtgw2.o
# 	rm -f dtgw2.o
# dtgw.so: dtgw.c
# 	g++ -c -fpic -Wall -fopenmp -std=c++17 -mavx2 -ftree-vectorize -march=haswell -maes -mno-pku -mno-sgx --param l1-cache-line-size=64 --param l1-cache-size=32 --param l2-cache-size=33792 -mabm dtgw2.c
# # 	g++ -c -fpic -Wall -std=c++17 -mavx2 dtgw.c
# 	g++ -shared -o build/dtgw.so dtgw2.o
# 	rm -f dtgw2.o
dtgw.so: dtgw.c
	g++ -c -fpic -Wall -std=c++17 -mavx2 -ftree-vectorize -march=haswell -maes -mno-pku -mno-sgx --param l1-cache-line-size=64 --param l1-cache-size=32 --param l2-cache-size=33792 dtgw2.c
# 	g++ -c -fpic -Wall -std=c++17 -mavx2 dtgw.c
	g++ -shared -o build/dtgw.so dtgw2.o
	rm -f dtgw2.o

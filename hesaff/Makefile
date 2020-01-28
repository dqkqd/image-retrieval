all: *.cpp
	g++ -O3 -Wall -o hesaff vlad.cpp pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp hesaff.cpp `pkg-config opencv --cflags --libs` -lrt 
	g++ -O3 -Wall -o extract_sift_from_dataset vlad.cpp pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp extract_sift_from_dataset.cpp `pkg-config opencv --cflags --libs` -lrt 
	g++ -O3 -Wall -o extract_sift_from_image vlad.cpp pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp extract_sift_from_image.cpp `pkg-config opencv --cflags --libs` -lrt 
	g++ -O3 -Wall -o extract_sift_from_image_quiet vlad.cpp pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp extract_sift_from_image_quiet.cpp `pkg-config opencv --cflags --libs` -lrt 
	g++ -O3 -Wall -o encode_dataset vlad.cpp pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp encode_dataset.cpp `pkg-config opencv --cflags --libs` -lrt 



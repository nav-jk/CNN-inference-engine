.PHONY : all clean build

build : 
	mkdir build
	build/main
	g++ main.c -o main

clean : 
	rm -rf build
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude -MMD -MP

SRC    := main.cpp $(wildcard src/*.cpp src/*/*.cpp)
OBJ    := $(patsubst %.cpp,build/%.o,$(SRC))
DEP    := $(OBJ:.o=.d)
TARGET := build/main

.PHONY : all run clean

all : $(TARGET)

$(TARGET) : $(OBJ)
	$(CXX) $(OBJ) -o $@

build/%.o : %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run : $(TARGET)
	./$(TARGET) weights.bin

clean :
	rm -rf build

-include $(DEP)

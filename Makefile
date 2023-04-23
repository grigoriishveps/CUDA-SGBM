CUDA_PATH ?= /usr/local/cuda

# CC = /usr/local/cuda/bin/nvcc -ccbin g++
CC = /usr/local/cuda/bin/nvcc
PROJECT = forrun
SRC = src/mainDiplom.cu
LIBS = `pkg-config --cflags --libs opencv4`
GENCODE_FLAGS = -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

SRC_DIR = src
OBJ_DIR = bin
# OBJS = $(OBJ_DIR)/mainDiplom.o $(OBJ_DIR)/helper.o
OBJS = $(OBJ_DIR)/mainDiplom.o


# preClear : 
# 	$(RM) $(PROJECT)

# OLD 
# Create Exutable file from oone file:
# $(PROJECT) : $(SRC)
# 	$(CC) $(GENCODE_FLAGS) -o $@ $+ $(LIBS) 

clean : $(PROJECT)
	$(RM) -r bin/* *.o

# Link c++ and CUDA compiled object files to target executable:
# $(PROJECT) : bin/mainDiplom.o
# 	$(CC) $(OBJS) $(GENCODE_FLAGS) -o $@ $(LIBS)

$(PROJECT) : mainDiplom.o
	$(CC) mainDiplom.o helper.o calc_cost.o calc_disparity.o old_calc_direction.o calc_path.o $(GENCODE_FLAGS) -o $@ $(LIBS)

# Compile main .cpp file to object files:
# $(OBJ_DIR)/%.o : %.cpp
# 	$(CC) $(CC_FLAGS) $(GENCODE_FLAGS) -c $< -o $@ 

# Compile C++ source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
# 	$(CC) $(CC_FLAGS) $(GENCODE_FLAGS) -c $< -o $@

# bin/mainDiplom.o : src/mainDiplom.cu bin/helpers/helper.o
# 	$(CC) $(GENCODE_FLAGS) -c $< -o $@ $(LIBS)

# Compile CUDA source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
# 	$(CC) $(GENCODE_FLAGS) -c $< -o $@ $+ $(LIBS)

# Compile CUDA source files to object files2:
# bin/helpers/helper.o : src/helpers/helper.cu src/helpers/helper.cuh
# 	$(CC) $(GENCODE_FLAGS) -c $< -o $@ $(LIBS) 

# bin/mainDiplom.o : src/mainDiplom.cu
# 	$(CC) $(GENCODE_FLAGS) -c $< -o $@ $+ $(LIBS)



# ---- nvidea 

mainDiplom.o : src/mainDiplom.cu src/helpers/helper.cu
	${CC} --gpu-architecture=sm_50 --device-c src/mainDiplom.cu src/helpers/helper.cu src/calc_cost/calc_cost.cu src/calc_disparity/calc_disparity.cu src/old_calc_direction/old_calc_direction.cu src/calc_path/calc_path.cu $(LIBS)


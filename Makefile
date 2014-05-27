CUDA_DIR=/usr/local/cuda-5.5
#CUDA_DIR=/opt/cuda
CUDA_SDK_DIR=/opt/cuda/sdk

#INCFLAG=-I$(CUDA_DIR)/include -I$(CUDA_SDK_DIR)/C/common/inc -I..
INCFLAG=-I$(CUDA_DIR)/include -I.. #`xml2-config --cflags`

#LIBFLAG=`xml2-config --libs`

DEBUG_FLAG=-g# -G
PTX_FLAG=-ptx

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   :=  $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) #$(GENCODE_SM13)  $(GENCODE_SM10)

NVCC = nvcc
NVCCFLAGS = $(INCFLAG) -use_fast_math $(GENCODE_FLAGS) $(DEBUG_FLAG) #-O3 #--ptxas-options=-v
#NVCCFLAGS = $(INCFLAG) $(GENCODE_SM30) $(DEBUG_FLAG) # -DTHRUST_DEBUG

#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM13) #use to compile on l2gpu with GTX285
#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM30)

#LIBS=-L/home/wittich/src/cudpp_install_2.0/lib -lcudpp

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: svt_gpu 

#svt_gpu: svt_gpu.o gf_unpack_cuda.o gf_unpack_thrust.o gf_fep.o gf_fit.o svtsim_functions.o
svt_gpu: svt_gpu.o gf_unpack_cuda.o gf_fep.o gf_fit.o svtsim_functions.o
	$(NVCC) $^ -o $@ $(LIBFLAG)

s2: s2.o
	$(NVCC) $^ -o $@ 


clean:
	$(RM) *.o *~ svt_gpu

depend:
	makedepend -Y $(INCFLAG) *.cu *.cc *.h

# DO NOT DELETE


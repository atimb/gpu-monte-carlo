################################################################################
#
# Build script for project scintillator
#
################################################################################


# Modify architecture accordingly to your GPU capabilities (sm_10, sm_11, sm_13)
SMVERSIONFLAGS= -arch sm_11
host_compilation_option := c++

# Add source files here
EXECUTABLE	:= ../../../projects/scintillator/scintillator
# CUDA source files (compiled with cudacc)
CUFILES		:= scintillator.cu
# CUDA dependency files
CU_DEPS		:= scintillator_kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= scintillator_gold.cpp


################################################################################
# Rules and targets

include ../../common/common.mk

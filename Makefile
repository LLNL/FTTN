# Compiler and flags
NVCC = nvcc
CUDA_ARCH = 86
# AMP_ARCH = gfx90a
HIPCC = hipcc
CFLAGS = --std=c++17 -I./utility/ 
LFLAGS = 
# rocWMMAPath = /people/lixi145/rocWMMA/library/include
rocWMMAPath = $(shell pwd)/rocWMMA/library/include


OUTPUT_NVIDIA = out_nvidia_
OUTPUT_AMD = out_amd_
# Object files and target
# SRC = fp64in.cpp
# TARGET_NVIDIA = tests_nvidia
# TARGET_AMD = tests_amd

# Default target
# all: $(TARGET_NVIDIA)
TESTS = fp16in bf16in tf-fp32in fp64in
# Compile for NVIDIA

# nvidia: CFLAGS += -DNVIDIA -arch=sm_$(CUDA_ARCH) 
# nvidia: $(TARGET_NVIDIA)
nvidia: $(addsuffix _NVIDIA, $(TESTS))

# Compile for AMD
# amd: $(rocWMMAPath)
$(rocWMMAPath): rocWMMA
# amd: CFLAGS += -DAMD -I$(rocWMMAPath)  #--amdgpu-target=$(AMP_ARCH) --offload-arch=$(AMP_ARCH)
# amd: $(TARGET_AMD)
amd: $(addsuffix _AMD, $(TESTS))

rocWMMA:
	git clone https://github.com/ROCm/rocWMMA.git

%_NVIDIA: %.cpp
	$(NVCC) $(CFLAGS) -DNVIDIA -arch=sm_$(CUDA_ARCH) $(LFLAGS) -x cu $< -o $(OUTPUT_NVIDIA)$*  
	./$(OUTPUT_NVIDIA)$* > $*_resultNVIDIA.txt

%_AMD: %.cpp $(rocWMMAPath) 
	$(HIPCC) $(CFLAGS) -DAMD -I$(rocWMMAPath) $(LFLAGS) $< -o $(OUTPUT_AMD)$*  
	./$(OUTPUT_AMD)$* > $*_resultAMD.txt

# Clean target
clean: 
	rm -f $(addprefix $(OUTPUT_NVIDIA), $(TESTS))
	rm -f $(addprefix $(OUTPUT_AMD), $(TESTS))
deep_clean:
	rm -f $(addprefix $(OUTPUT_NVIDIA), $(TESTS))
	rm -f $(addprefix $(OUTPUT_AMD), $(TESTS))
	rm -f *_resultNVIDIA.txt
	rm -f *_resultAMD.txt
	rm -rf ./rocWMMA

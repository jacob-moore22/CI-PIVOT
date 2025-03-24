TARGET	:= StreamTriadCuda

CC_SRCS	:= timer.cc
CU_SRCS	:= StreamTriad.cu
CU_SRCS_V2:= StreamTriad_v2.cu

OBJS		:= $(CC_SRCS:%.cc=%.o) $(CU_SRCS:%.cu=%.o)
DEPS		:= $(CC_SRCS:%.cc=%.d) $(CU_SRCS:%.cu=%.d)

OBJS_V2		:= $(CC_SRCS:%.cc=%.o) $(CU_SRCS_V2:%.cu=%.o)
DEPS_V2		:= $(CC_SRCS:%.cc=%.d) $(CU_SRCS_V2:%.cu=%.d)

# Get the compute capability.
ARCH!= nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/\.//g'

NVCC:= nvcc
NVCC_FLAGS:= -O3 -arch=sm_$(ARCH)

default: arch $(TARGET) $(TARGET)_v2
	$(CURDIR)/$(TARGET)
	$(CURDIR)/$(TARGET)_v2

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(TARGET)_v2: $(OBJS_V2)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

arch:
	@echo "Compute capability: $(ARCH)"

%.o: %.cc
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.d: %.cc
	$(NVCC) $(NVCC_FLAGS) -M $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.d: %.cu
	$(NVCC) $(NVCC_FLAGS) -M $< -o $@

clean:
	rm -rf $(OBJS) $(DEPS) $(TARGET) *~
	rm -rf $(OBJS_V2) $(DEPS_V2) $(TARGET)_v2 *~

.PHONY: arch

-include $(DEPS)
-include $(DEPS_V2)

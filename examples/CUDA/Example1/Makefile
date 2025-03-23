TARGET:= StreamTriadCuda

CC_SRCS:= timer.cc
CU_SRCS:= StreamTriad.cu
OBJS:= $(CC_SRCS:%.cc=%.o) $(CU_SRCS:%.cu=%.o)
DEPS:= $(CC_SRCS:%.cc=%.d) $(CU_SRCS:%.cu=%.d)

NVCC:= nvcc
# Get the compute capability.
ARCH!= nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/\.//g'

NVCC_FLAGS:= -O3 -arch=sm_$(ARCH)

default: arch $(TARGET)
	$(CURDIR)/$(TARGET)

$(TARGET): $(OBJS)
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

.PHONY: arch

-include $(DEPS)

TARGET:= StreamTriadCuda

CC_SRCS:= timer.cc
CU_SRCS:= StreamTriad.cu
OBJS:= $(CC_SRCS:%.cc=%.o) $(CU_SRCS:%.cu=%.o)
DEPS:= $(CC_SRCS:%.cc=%.d) $(CU_SRCS:%.cu=%.d)

NVCC:= nvcc
NVCC_FLAGS:= -O3 -arch=sm_80

default: $(TARGET)
	$(CURDIR)/$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

%.o: %.cc
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.d: %.cc
	$(NVCC) -M $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.d: %.cu
	$(NVCC) -M $< -o $@

clean:
	rm -rf $(OBJS) $(DEPS) $(TARGET) *~

-include $(DEPS)

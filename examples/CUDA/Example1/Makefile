TARGET:= foo

CC_SRCS:= timer.cc
CU_SRCS:= StreamTriad.cu
OBJS:= $(CC_SRCS:%.cc=%.o) $(CU_SRCS:%.cu=%.o)
DEPS:= $(CC_SRCS:%.cc=%.d) $(CU_SRCS:%.cu=%.d)

NVCC:= nvcc

default: $(TARGET)
	$(CURDIR)/$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $^ -o $@

%.o: %.cc
	$(NVCC) -c $< -o $@

%.d: %.cc
	$(NVCC) -M $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

%.d: %.cu
	$(NVCC) -M $< -o $@

clean:
	rm -rf $(OBJS) $(DEPS) $(TARGET) *~

-include $(DEPS)

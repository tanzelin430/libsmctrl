CC = gcc
CXX = g++
NVCC ?= nvcc
# -fPIC is needed in all cases, as we may be linked into another shared library
CFLAGS = -fPIC
LDFLAGS = -lcuda -I/usr/local/cuda/include -ldl

.PHONY: clean tests

libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ $(CFLAGS) $(LDFLAGS)

libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# Use static linking with tests to avoid LD_LIBRARY_PATH issues
libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c libsmctrl.a testbench.h
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_mask_shared.o: libsmctrl_test_mask_shared.cu testbench.h
	$(NVCC) -ccbin $(CXX) $< -c -g

libsmctrl_test_global_mask: libsmctrl_test_global_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask: libsmctrl_test_stream_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask_override: libsmctrl_test_stream_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask: libsmctrl_test_next_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask_override: libsmctrl_test_next_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

tests: libsmctrl_test_gpc_info libsmctrl_test_global_mask libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override libsmctrl_test_next_mask libsmctrl_test_next_mask_override

clean:
	rm -f libsmctrl.so libsmctrl.a libsmctrl_test_gpu_info \
	      libsmctrl_test_mask_shared.o libmsctrl_test_global_mask \
	      libsmctrl_test_stream_mask libmsctrl_test_stream_mask_override \
	      libsmctrl_test_next_mask libmsctrl_test_next_mask_override

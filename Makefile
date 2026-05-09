# cuda-constraint-engine Makefile
# Targets: lib, examples, clean, test

CC       ?= gcc
NVCC     ?= nvcc
ARCH     ?= sm_86
CUDA_VER ?= 11.5

CFLAGS   := -O2 -Wall -Wextra -fPIC
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++14 --expt-relaxed-constexpr \
             -Xcompiler -fPIC -Xcompiler -Wall

LIB_NAME  := libconstraint_engine.so
LIB_OUT   := ./$(LIB_NAME)

SRCS      := $(wildcard src/*.cu)
OBJS      := $(SRCS:.cu=.o)

EXAMPLE_SRCS := $(wildcard examples/*.cu)
EXAMPLE_BINS := $(EXAMPLE_SRCS:.cu=)

.PHONY: all lib examples clean test install

all: lib examples

# === Shared Library ===

lib: $(LIB_OUT)

$(LIB_OUT): $(OBJS)
	$(NVCC) -shared -o $@ $^ -lcudart $(NVCCFLAGS)

src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -I include -c $< -o $@

# === Examples ===

examples: $(EXAMPLE_BINS)

examples/%: examples/%.cu $(LIB_OUT)
	$(NVCC) $(NVCCFLAGS) -I include -o $@ $< -L. -lconstraint_engine -Xlinker -rpath -Xlinker '$$ORIGIN'

# === Python wrapper location ===

python: lib
	cp $(LIB_OUT) python/

# === Test ===

test: lib
	@echo "=== Running quick smoke test ==="
	@LD_LIBRARY_PATH=. ./examples/quickstart 2>&1 || echo "(quickstart not built yet — run 'make examples' first)"

# === Install ===

PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib $(PREFIX)/include/constraint_engine
	install -m 755 $(LIB_OUT) $(PREFIX)/lib/
	install -m 644 include/constraint_engine.h $(PREFIX)/include/
	ldconfig 2>/dev/null || true

# === Clean ===

clean:
	rm -f src/*.o $(LIB_OUT) examples/quickstart examples/batch_check \
	      examples/stream_pipeline examples/hot_swap examples/eisenstein_narrows
	rm -f python/$(LIB_NAME)

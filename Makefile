BUILD=build
CC=g++

normal: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -o $(BUILD)/$@

blocking: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_BLOCKING -o $(BUILD)/$@

prefetch: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_PREFETCH -o $(BUILD)/$@

simd: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_SIMD -o $(BUILD)/$@

blocking-prefetch: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_BLOCKING_PREFETCH -o $(BUILD)/$@

blocking-simd: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_BLOCKING_SIMD -o $(BUILD)/$@

simd-prefetch: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_SIMD_PREFETCH -o $(BUILD)/$@

blocking-simd-prefetch: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

all: build
	g++ -mavx512f -mavx512dq pa1-the-matrix.c -D OPTIMIZE_BLOCKING -D OPTIMIZE_SIMD -D OPTIMIZE_PREFETCH -D OPTIMIZE_BLOCKING_PREFETCH -D OPTIMIZE_BLOCKING_SIMD -D OPTIMIZE_SIMD_PREFETCH -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

clean:
	@rm -rf $(BUILD)
	@rm -f out.txt

build:
	@mkdir -p $(BUILD)

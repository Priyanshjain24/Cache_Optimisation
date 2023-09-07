# The-Matrix

Performed matrix multiplication using various optimization techniques and their combinations:
- blocked matrix multiplication
- SIMD instructions
- software prefetching

and combination of these techniques.

<br>

---
## Task 1: Blocked Matrix Multiplication

<br>

Blocking improves matrix multiplication performance by optimizing cache usage, reducing cache misses, and enhancing temporal locality. It achieves this by breaking matrices into smaller blocks that fit well in L1 and L2 caches. Processing one block at a time ensures repeated access to the same block data, further enhancing temporal locality. Steps involved:

1. **Divide matrices into blocks**:
   - Use two outer loops to go through specific blocks of the resulting matrix.
   - Employ another outer loop to go through all blocks of input matrices that contribute to the resulting matrix.

2. **Block-Wise Matrix Multiplication**:
   - Inside the resulting matrix's block, utilize two inner loops to process individual elements.
   - Employ an additional inner loop to traverse through all rows and columns of input matrices for block-wise matrix multiplication.

| Matrix Size | Performance | 
|-------------|-------------|
| 100         |    1.60     | 
| 200         |    1.40     | 
| 800         |    1.15     |

---
## Task 2: SIMD instructions

<br>

SIMD (Single Instruction, Multiple Data) instructions are specialized instructions that can perform operations on multiple data elements simultaneously. For our hardware, we found two types of SIMD instructions, namely SSE (Streaming SIMD Extensions) and AVX (Advanced Vector Extensions). We used AVX as we can process 512 bits parallely over 128 bits for SIMD, along with some more instructions like FMA (Fused Multiplication Addition). We can use SIMD instructions for matrix multiplication by doing FMA on 8 elements from rows and coloumns of input matrices parallely. Steps Involved:

1. **Initialize a SIMD Register to Zero**:
   - Initializes a 512-bit wide SIMD register to zero which stores the results of the matrix multiplication.
     
2. **Loop over rows and coloumns of result matrix**

3. **SIMD Parallelization with AVX-512**:
   - A third loop iterates over rows and coloumns of input matrices that contribute to a particular element of the result matrix.
   - We load 8 doubles from row of first matrix and coloumn of second matrix into 2 different SIMD registers.
   - Perform FMA operation on the above two SIMD register and the sum SIMD register.

4. **Store SIMD Accumulated Result in an array**

5. **Sum the array values and store the result in a variable**

6. **Handle Remaining Values**:
   - Since the matrix dimension may not be a multiple of 8, we need to handle the calculations of the remaining terms seperately.
   - For these, we perform computation like normal matrix multiplication and increment the resut variable.
     
7. **Store the answer in the resulting matrix**

| Matrix Size | Performance | 
|-------------|-------------|
| 100         |     5.00    | 
| 200         |     4.50    | 
| 800         |     4.75    |

---
## Task 3: Software Prefetching

<br>

Software prefetching enhances matrix multiplication by proactively fetching data from memory before it's needed, reducing memory access latency. This optimization boosts cache utilization, reducing cache misses and improving CPU efficiency. It also facilitates parallelism by overlapping data loading and computation, harnessing CPU resources more effectively. The multiplication here involved traversing in ijk loop format.
Steps involved:

1. **Prefetching initial values**
   The first intial address locations of A,B,C are fetched for every iteration of ith Loop in the beginning and then the loop execution is started.

2. **Fetch A block and B matrix**
   The jth loop is executed for j=0 outside the loop and values of A & B matrices are fetched then. Then the inner (j,k) loops are executed with the already fetched values of A and B matrices and multiplication is carried out then.

| Matrix Size | Performance | 
|-------------|-------------|
| 100         |     2.00    | 
| 200         |     1.55    | 
| 800         |     1.48    |

---
## Task 4: Blocked Matrix Multiplication + Software Prefetching

<br>

| Matrix Size | Performance | 
|-------------|-------------|
| 100         |    1.62     | 
| 200         |    1.21     | 
| 800         |    1.29     |

---
## Task 5: SIMD instructions + Software Prefetching

<br>

| Matrix Size | Performance | 
|-------------|-------------|
| 100         |    5.00     | 
| 200         |    4.10     | 
| 800         |    4.00     |

---
## Observations

Blocked Matrix Multiplication improves cache efficiency but requires careful block size selection. SIMD Instructions leverage parallelism but depend on proper data alignment. Software Prefetching reduces cache misses but needs precise tuning. Combinations like Blocking + SIMD or Prefetching offer substantial gains but introduce complexity. Effective optimization depends on understanding hardware, data patterns, and careful parameter tuning for optimal performance.

Knowing hardware details is vital for peak performance. Things like cache size, memory hierarchy, and vectorization impact how we design and optimize algorithms. Neglecting them leads to inefficiencies, slower memory access, and performance bottlenecks. To make the most of parallelism, reduce memory delays, and boost cache efficiency, software should align with hardware specifics, ensuring faster execution and better processor use.

![Performance Comparision](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-paradox_bits/blob/master/image.png)

<br>

---
<!-- All the best! :smile: -->

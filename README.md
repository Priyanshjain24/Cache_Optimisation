[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/mnOJa0WY)
# PA1-The-Matrix

Perform matrix multiplication using various optimization techniques:
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

<style>
  table {
    border-collapse: collapse;
    width: 100%;
    text-align: center; /* Center align all text within the table */
  }
  th, td {
    border: 1px solid black;
    text-align: center; /* Center align text within table cells */
    padding: 8px;
  }
</style>

<table>
<caption><b>Execution Time for Blocking</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matirx (s)</th>
    <th colspan="2">Time for size 200 matirx (s)</th>
    <th colspan="2">Time for size 800 matirx (s)</th>
  </tr>
  <tr>
    <th>Blocked</th>
    <th>Normal</th>
    <th>Blocked</th>
    <th>Normal</th>
    <th>Blocked</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.007799</td>
    <td>0.011913</td>
    <td>0.027335</td>
    <td>0.028027</td>
    <td>1.695288</td>
    <td>1.996637</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.004427</td>
    <td>0.006373</td>
    <td>0.028393</td>
    <td>0.031876</td>
    <td>1.655949</td>
    <td>1.923453</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.003660</td>
    <td>0.006411</td>
    <td>0.027083</td>
    <td>0.038172</td>
    <td>1.700456</td>
    <td>2.024593</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.005295</td>
    <td>0.008232</td>
    <td>0.027604</td>
    <td>0.032692</td>
    <td>1.683898</td>
    <td>1.981561</td>
  </tr>
</table>

On average the blocked matrix multiplication had a speedup of __1.55__, __1.18__ and __1.17__ for matrices with size __100__, __200__ and __800__ respectively.

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


<!-- This is a hidden text -->
<style>
  table {
    border-collapse: collapse;
    width: 100%;
    text-align: center; /* Center align all text within the table */
  }
  th, td {
    border: 1px solid black;
    text-align: center; /* Center align text within table cells */
    padding: 8px;
  }
</style>
<!-- This is a hidden text -->

<table>
<caption><b>Execution Time for SIMD</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matirx (s)</th>
    <th colspan="2">Time for size 200 matirx (s)</th>
    <th colspan="2">Time for size 800 matirx (s)</th>
  </tr>
  <tr>
    <th>SIMD</th>
    <th>Normal</th>
    <th>SIMD</th>
    <th>Normal</th>
    <th>SIMD</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.007799</td>
    <td>0.011913</td>
    <td>0.027335</td>
    <td>0.028027</td>
    <td>1.695288</td>
    <td>1.996637</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.004427</td>
    <td>0.006373</td>
    <td>0.028393</td>
    <td>0.031876</td>
    <td>1.655949</td>
    <td>1.923453</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.003660</td>
    <td>0.006411</td>
    <td>0.027083</td>
    <td>0.038172</td>
    <td>1.700456</td>
    <td>2.024593</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.005295</td>
    <td>0.008232</td>
    <td>0.027604</td>
    <td>0.032692</td>
    <td>1.683898</td>
    <td>1.981561</td>
  </tr>
</table>

On average the SIMD matrix multiplication had a speedup of __1.55__, __1.18__ and __1.17__ for matrices with size __100__, __200__ and __800__ respectively.


---
## Task 3: Software Prefetching

<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions

<!-- This is a hidden text -->
<style>
  table {
    border-collapse: collapse;
    width: 100%;
    text-align: center; /* Center align all text within the table */
  }
  th, td {
    border: 1px solid black;
    text-align: center; /* Center align text within table cells */
    padding: 8px;
  }
</style>
<!-- This is a hidden text -->

<table>
<caption><b>Execution Time for Blocking</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matirx (s)</th>
    <th colspan="2">Time for size 200 matirx (s)</th>
    <th colspan="2">Time for size 800 matirx (s)</th>
  </tr>
  <tr>
    <th>Blocked-SIMD</th>
    <th>Normal</th>
    <th>Blocked-SIMD</th>
    <th>Normal</th>
    <th>Blocked-SIMD</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.007799</td>
    <td>0.011913</td>
    <td>0.027335</td>
    <td>0.028027</td>
    <td>1.695288</td>
    <td>1.996637</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.004427</td>
    <td>0.006373</td>
    <td>0.028393</td>
    <td>0.031876</td>
    <td>1.655949</td>
    <td>1.923453</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.003660</td>
    <td>0.006411</td>
    <td>0.027083</td>
    <td>0.038172</td>
    <td>1.700456</td>
    <td>2.024593</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.005295</td>
    <td>0.008232</td>
    <td>0.027604</td>
    <td>0.032692</td>
    <td>1.683898</td>
    <td>1.981561</td>
  </tr>
</table>

On average the blocked-SIMD matrix multiplication had a speedup of __1.55__, __1.18__ and __1.17__ for matrices with size __100__, __200__ and __800__ respectively.

<br>

---
## Bonus Task 2: Blocked Matrix Multiplication + Software Prefetching

<br>

---
## Bonus Task 3: SIMD instructions + Software Prefetching

<br>

---
## Bonus Task 4: Bloced Matrix Multiplication + SIMD instructions + Software Prefetching

<br>

---
All the best! :smile:

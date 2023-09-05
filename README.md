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
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
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
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
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
    <td>0.002011</td>
    <td>0.005602</td>
    <td>0.005057</td>
    <td>0.028209</td>
    <td>0.384643</td>
    <td>1.943984</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.002590</td>
    <td>0.011588</td>
    <td>0.005048</td>
    <td>0.030393</td>
    <td>0.387657</td>
    <td>2.063484</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.000830</td>
    <td>0.005964</td>
    <td>0.004923</td>
    <td>0.038690</td>
    <td>0.399986</td>
    <td>1.973076</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.001810</td>
    <td>0.007718</td>
    <td>0.005009</td>
    <td>0.032430</td>
    <td>0.390762</td>
    <td>1.993514</td>
  </tr>
</table>

On average the SIMD matrix multiplication had a speedup of __4.26__, __6.47__ and __5.10__ for matrices with size __100__, __200__ and __800__ respectively.


---
## Task 3: Software Prefetching

<br>

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
<caption><b>Execution Time for Prefetching</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
  </tr>
  <tr>
    <th>Prefetch</th>
    <th>Normal</th>
    <th>Prefetch</th>
    <th>Normal</th>
    <th>Prefetch</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.002904</td>
    <td>0.003207</td>
    <td>0.024199</td>
    <td>0.028171</td>
    <td>1.792711</td>
    <td>1.975869</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.003236</td>
    <td>0.005929</td>
    <td>0.023613</td>
    <td>0.030164</td>
    <td>1.624470</td>
    <td>1.975095</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.003165</td>
    <td>0.005741</td>
    <td>0.024629</td>
    <td>0.030771</td>
    <td>1.672661</td>
    <td>1.989801</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.003101</td>
    <td>0.004959</td>
    <td>0.024147</td>
    <td>0.029702</td>
    <td>1.696614</td>
    <td>1.980257</td>
  </tr>
</table>

On average Software Prefetching had a speedup of __1.60__, __1.23__ and __1.16__ for matrices with size __100__, __200__ and __800__ respectively.

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
<caption><b>Execution Time for Blocking-SIMD</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
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
    <td>0.000953</td>
    <td>0.004298</td>
    <td>0.008040</td>
    <td>0.029656</td>
    <td>0.459524</td>
    <td>1.948890</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.000997</td>
    <td>0.006544</td>
    <td>0.007302</td>
    <td>0.027809</td>
    <td>0.467449</td>
    <td>1.970119</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.001030</td>
    <td>0.005417</td>
    <td>0.008853</td>
    <td>0.032062</td>
    <td>0.460823</td>
    <td>2.094328</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.000993</td>
    <td>0.005419</td>
    <td>0.008065</td>
    <td>0.029842</td>
    <td>0.462598</td>
    <td>2.037779</td>
  </tr>
</table>

On average the blocked-SIMD matrix multiplication had a speedup of __5.45__, __3.70__ and __4.40__ for matrices with size __100__, __200__ and __800__ respectively.

<br>

---
## Bonus Task 2: Blocked Matrix Multiplication + Software Prefetching

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
<caption><b>Execution Time for Blocking-Prefetching</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
  </tr>
  <tr>
    <th>Blocked-Prefetch</th>
    <th>Normal</th>
    <th>Blocked-Prefetch</th>
    <th>Normal</th>
    <th>Blocked-Prefetch</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.003631</td>
    <td>0.004015</td>
    <td>0.026740</td>
    <td>0.027444</td>
    <td>1.756630</td>
    <td>1.987905</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.003643</td>
    <td>0.006039</td>
    <td>0.026810</td>
    <td>0.027418</td>
    <td>1.711984</td>
    <td>2.036989</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.004579</td>
    <td>0.005799</td>
    <td>0.028126</td>
    <td>0.031117</td>
    <td>1.718862</td>
    <td>1.997806</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.003951</td>
    <td>0.005284</td>
    <td>0.027225</td>
    <td>0.028659</td>
    <td>1.729158</td>
    <td>2.007566</td>
  </tr>
</table>

On average the blocked software prefetching matrix multiplication had a speedup of __1.34__, __1.05__ and __1.16__ for matrices with size __100__, __200__ and __800__ respectively.

<br>

---
## Bonus Task 3: SIMD instructions + Software Prefetching

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
<caption><b>Execution Time for SIMD-Prefetching</b></caption>
  <tr>
    <th rowspan="2">Test No.</th>
    <th colspan="2">Time for size 100 matrix (s)</th>
    <th colspan="2">Time for size 200 matrix (s)</th>
    <th colspan="2">Time for size 800 matrix (s)</th>
  </tr>
  <tr>
    <th>SIMD-Prefetch</th>
    <th>Normal</th>
    <th>SIMD-Prefetch</th>
    <th>Normal</th>
    <th>SIMD-Prefetch</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td>Attempt 1</td>
    <td>0.000997</td>
    <td>0.006073</td>
    <td>0.005923</td>
    <td>0.031766</td>
    <td>0.431522</td>
    <td>1.977474</td>
  </tr>
  <tr>
    <td>Attempt 2</td>
    <td>0.000912</td>
    <td>0.005889</td>
    <td>0.005770</td>
    <td>0.031696</td>
    <td>0.433507</td>
    <td>2.084508</td>
  </tr>
  <tr>
    <td>Attempt 3</td>
    <td>0.002011</td>
    <td>0.010471</td>
    <td>0.006085</td>
    <td>0.028013</td>
    <td>0.430351</td>
    <td>1.947223</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.001306</td>
    <td>0.007477</td>
    <td>0.005926</td>
    <td>0.030491</td>
    <td>0.431793</td>
    <td>2.003068</td>
  </tr>
</table>

On average the SIMD software prefetching matrix multiplication had a speedup of __5.72__, __5.14__ and __4.64__ for matrices with size __100__, __200__ and __800__ respectively.

<br>

---
## Bonus Task 4: Bloced Matrix Multiplication + SIMD instructions + Software Prefetching

<br>

---
## Results

<table>
  <thead>
    <tr>
      <th>Matrix Size</th>
      <th>Blocking</th>
      <th>SIMD</th>
      <th>Prefetch</th>
      <th>Blocked-SIMD</th>
      <th>BLocked-Prefetch</th>
      <th>SIMD-Prefetch</th>
      <th>Blocked-SIMD-Prefetch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>Row 1, Col 2</td>
      <td>Row 1, Col 3</td>
      <td>Row 1, Col 4</td>
      <td>Row 1, Col 5</td>
      <td>Row 1, Col 6</td>
      <td>Row 1, Col 7</td>
      <td>Row 1, Col 8</td>
    </tr>
    <tr>
      <td>200</td>
      <td>Row 2, Col 2</td>
      <td>Row 2, Col 3</td>
      <td>Row 2, Col 4</td>
      <td>Row 2, Col 5</td>
      <td>Row 2, Col 6</td>
      <td>Row 2, Col 7</td>
      <td>Row 2, Col 8</td>
    </tr>
    <tr>
      <td>800</td>
      <td>Row 3, Col 2</td>
      <td>Row 3, Col 3</td>
      <td>Row 3, Col 4</td>
      <td>Row 3, Col 5</td>
      <td>Row 3, Col 6</td>
      <td>Row 3, Col 7</td>
      <td>Row 3, Col 8</td>
    </tr>
  </tbody>
</table>


<br>

---
<!-- All the best! :smile: -->

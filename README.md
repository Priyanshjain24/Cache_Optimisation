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

```html
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
<caption><b>Execution Time Table</b></caption>
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
```

On average the blocked matrix multiplication had a speedup of __1.55__, __1.18__ and __1.17__ for matrices with size __100__, __200__ and __800__ respectively.

---
## Task 2: SIMD instructions

<br>

---
## Task 3: Software Prefetching

<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions

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

# **📌 Empirical Analysis of Fibonacci Algorithms**

## **📖 Overview**

This study explores and compares different algorithms for computing the **Fibonacci sequence**. The goal is to analyze the **efficiency, scalability, and execution time** of multiple approaches ranging from naïve recursion to matrix exponentiation.

The Fibonacci sequence is defined as:
$$
F(n) =
\begin{cases}
    0, & n = 0 \\
    1, & n = 1 \\
    F(n-1) + F(n-2), & n \geq 2
\end{cases}
$$

Various computational methods are implemented and evaluated to determine their suitability for different problem sizes.

## **🎯 Objectives**

- Implement and compare six different Fibonacci algorithms.
- Measure and analyze their empirical performance.
- Identify the most efficient method for varying input sizes.
- Visualize execution time trends through graphs.

## **📂 Project Structure**

```
📂 01-fibonacci/
│── 📜 README.md                   # This document
│── 📜 01-fibonacci.ipynb          # Jupyter Notebook implementation
│── 📂 result/                     # Performance results and visualizations
│   ├── 📊 fibonacci_results.csv   # Execution time results
│   ├── 📂 plots/                  # Graphs for performance analysis
```

## **🛠 Implemented Fibonacci Methods**

### **1️⃣ Naïve Recursive Approach** *(Exponential $O(2^n)$)*

- Implements the basic recursive definition of Fibonacci.
- **Highly inefficient** for large $n$ due to redundant computations.

### **2️⃣ Recursive with Memoization** *(Top-Down DP, $O(n)$)*

- Uses a **cache (dictionary)** to store previously computed values.
- **Reduces redundant calculations** significantly.
- **Linear time complexity** $O(n)$.

### **3️⃣ Iterative Dynamic Programming** *(Bottom-Up DP, $O(n)$)*

- Builds the Fibonacci sequence **iteratively** in an array.
- Eliminates recursion overhead but requires **$O(n)$ space**.

### **4️⃣ Matrix Exponentiation** *(Logarithmic $O(\log n)$)*

- Uses **matrix multiplication** to compute Fibonacci numbers efficiently.
- **Best method for very large values of $n$**.
- **Logarithmic time complexity** $O(\log n)$.

### **5️⃣ Binet’s Formula** *(Constant $O(1)$)*

- Uses the **Golden Ratio (Phi)** for a closed-form solution.
- **Extremely fast but unreliable** for large $n$ due to floating-point precision errors.

### **6️⃣ Optimized Iterative Approach** *(Linear $O(n)$, Constant Space $O(1)$)*

- **Stores only the last two computed values**, reducing memory usage.
- **Most memory-efficient iterative approach**.

## **🔬 Experimental Setup**

### **📌 Input Data**

- **Small Inputs**: $n = \{5, 7, 10, 12, 15, 18, 20, 25, 30, 50, 100\}$
- **Large Inputs**: $n = \{200, 500, 700, 1000, 3000, 5000, 7000, 10000, 15000, 20000\}$

### **📌 Performance Measurement**

- **Each algorithm** is executed **multiple times** for each input size.
- **Execution time** is recorded using Python’s `timeit` module.
- **Results are stored in a CSV file** (`fibonacci_results.csv`).
- **Graphs are generated** to visualize performance trends.

## **📌 Conclusion**

This study provides valuable insights into **choosing the right algorithm** for computing Fibonacci numbers. The findings include:

- **Recursive approach** is highly inefficient and **should be avoided** for large $n$.
- **Memoization & Iterative DP** provide a **linear-time solution** but require additional space.
- **Matrix Exponentiation** is the **fastest method for very large values of $n$**.
- **Binet’s Formula** is **only reliable for small $n$** due to floating-point errors.
- **Optimized Iterative $O(1)$ Space Method** provides a **good balance between efficiency and memory usage**.

### **🚀 Best Algorithm for Different Use Cases**

| **Use Case**                   | **Recommended Algorithm**           |
|--------------------------------|-------------------------------------|
| **Small $n \leq 30$**          | Recursive (for learning purposes)   |
| **Moderate $n < 10^5$**        | Memoization (Top-Down DP)           |
| **Large $n < 10^6$**           | Iterative DP or Optimized Iterative |
| **Very Large $n > 10^6$**      | Matrix Exponentiation               |
| **Quick Approximation Needed** | Binet’s Formula (for small $n$)     |

---

## **💾 Running the tests**

### **📥 Clone the Repository**

```bash
git clone https://github.com/sergedbs/algorithms-analysis.git
cd algorithms-analysis/notebooks/01-fibonacci
```

### **▶️ Run the Jupyter Notebook**

```bash
jupyter notebook 01-fibonacci.ipynb
```

### **📊 View Results**

- Check the **CSV file**: `result/fibonacci_results.csv`
- View **performance graphs**: `result/plots/`

---

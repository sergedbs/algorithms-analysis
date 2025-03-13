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

## **🛠 Implemented Fibonacci Methods**

### **1️⃣ Naïve Recursive Approach** *(Exponential $`O(2^n)`$)*

- Implements the basic recursive definition of Fibonacci.
- **Highly inefficient** for large $n$ due to redundant computations.

### **2️⃣ Recursive with Memoization** *(Top-Down DP, $`O(n)`$)*

- Uses a **cache (dictionary)** to store previously computed values.
- **Reduces redundant calculations** significantly.
- **Linear time complexity** $O(n)$.

### **3️⃣ Iterative Dynamic Programming** *(Bottom-Up DP, $`O(n)`$)*

- Builds the Fibonacci sequence **iteratively** in an array.
- Eliminates recursion overhead but requires **$O(n)$ space**.

### **4️⃣ Matrix Exponentiation** *(Logarithmic $`O(\log n)`$)*

- Uses **matrix multiplication** to compute Fibonacci numbers efficiently.
- **Best method for very large values of $n$**.
- **Logarithmic time complexity** $O(\log n)$.

### **5️⃣ Binet’s Formula** *(Constant $`O(1)`$)*

- Uses the **Golden Ratio (Phi)** for a closed-form solution.
- **Extremely fast but unreliable** for large $n$ due to floating-point precision errors.

### **6️⃣ Binet's Formula with Decimal Precision** *(Constant $`O(1)`$)*

- Uses the `decimal` module for higher precision arithmetic.
- **More reliable for large values of $n$**.

### **7️⃣ Optimized Iterative Approach** *(Linear $`O(n)`$, Constant Space $`O(1)`$)*

- **Stores only the last two computed values**, reducing memory usage.
- **Most memory-efficient iterative approach**.

### **8️⃣ Fast Doubling Method** *(Logarithmic $`O(\log n)`$)*

- Utilizes a **fast doubling technique** to compute Fibonacci numbers.
- **Achieves logarithmic time complexity** $O(\log n)$.

## **🔬 Experimental Setup**

### **📌 Input Data**

- **Small Inputs**: $n = \{1, 2, 3, \ldots, 198\}$
- **Large Inputs**: $n = \{200, 300, 400, \ldots, 20000\}$

### **📌 Performance Measurement**

- **Each algorithm** is executed **multiple times** for each input size.
- **Execution time** is recorded using Python’s `time` module.
- **Results are stored in a CSV file** (`results.csv`).
- **Graphs are generated** to visualize performance trends.

## **📌 Conclusion**

This study provides valuable insights into **choosing the right algorithm** for computing Fibonacci numbers. The findings include:

- **Recursive approach** is highly inefficient and **should be avoided** for large $n$.
- **Memoization & Iterative DP** provide a **linear-time solution** but require additional space.
- **Matrix Exponentiation** is the **fastest method for very large values of $n$**.
- **Binet’s Formula** is **only reliable for small $n$** due to floating-point errors.
- **Optimized Iterative $O(1)$ Space Method** provides a **good balance between efficiency and memory usage**.
- **Fast Doubling Method** is highly efficient for large $n$.

### **🚀 Best Algorithm for Different Use Cases**

| **Use Case**                   | **Recommended Algorithm**              |
|--------------------------------|----------------------------------------|
| **Small $n \leq 30$**          | Recursive (for learning purposes)      |
| **Moderate $n < 10^5$**        | Memoization (Top-Down DP)              |
| **Large $n < 10^6$**           | Iterative DP or Optimized Iterative    |
| **Very Large $n > 10^6$**      | Matrix Exponentiation or Fast Doubling |
| **Quick Approximation Needed** | Binet’s Formula (for small $n$)        |

---

## **💾 Running the tests**

### **📥 Clone the Repository**

```bash
git clone https://github.com/sergedbs/algorithms-analysis.git
cd algorithms-analysis
```

### **🛠 Install Dependencies** _(if not already installed)_
The dependencies can be installed using either method:

```bash
# Method 1: Using pip with requirements.txt
pip install -r requirements.txt

# Method 2: Install as a package (development mode)
pip install -e .
```

### **▶️ Run the Jupyter Notebook**

```bash
cd notebooks/01-fibonacci
jupyter notebook 01-fibonacci.ipynb
```

### **📊 View Results**

- Check the **CSV file**: `result/results.csv`
- View **performance graphs**: `result/plots/`

## **📂 Project Structure**

```plaintext
📂 01-fibonacci/
│── 📓 01-fibonacci.ipynb          # Jupyter Notebook implementation
│── 📂 result/                     # Performance results and visualizations
│   ├── 📊 results.csv             # Execution time results
│   └── 📂 plots/                  # Graphs for performance analysis
└── 📄 README.md                   # This file
```

> This study is part of the _**Empirical Analysis of Algorithms**_ repository. The complete source code and Jupyter notebooks are available on GitHub: [Empirical Analysis of Algorithms](https://github.com/sergedbs/algorithms-analysis).

---

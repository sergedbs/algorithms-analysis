# **Empirical Analysis of Algorithms**

This repository contains implementations and empirical analyses of various algorithms, focusing on performance evaluation and complexity analysis.

## **📚 Repository Contents**

- [**Fibonacci Algorithms**](notebooks/01-fibonacci): Comparing 8 different implementation methods
- [**Sorting Algorithms**](notebooks/02-sorting): Analysis of classic sorting techniques

## **🛠️ Getting Started**

### **📋 Prerequisites**

- Python 3.7 or higher
- Jupyter Notebook environment

### **📥 Clone the Repository**

```bash
git clone https://github.com/sergedbs/algorithms-analysis.git
cd algorithms-analysis
```

### **🛠 Install Dependencies**

The dependencies can be installed using either method:

```bash
# Method 1: Using pip with requirements.txt
pip install -r requirements.txt

# Method 2: Install as a package (development mode)
pip install -e .
```

## **📊 Available Studies**

### [**1. Fibonacci Algorithms Analysis**](notebooks/01-fibonacci)

Implements and compares 8 different methods for computing Fibonacci numbers:

- Naïve recursive approach
- Memoization
- Dynamic programming
- Matrix exponentiation
- Binet's formula (and decimal precision variant)
- Space-optimized iterative solution
- Fast doubling method

```bash
cd notebooks/01-fibonacci
jupyter notebook 01-fibonacci.ipynb
```

### [**2. Sorting Algorithms Analysis**](notebooks/02-sorting)

Implements and compares the performance of classic sorting algorithms:

- Quick Sort
- Merge Sort
- Heap Sort
- Insertion Sort

```bash
cd notebooks/02-sorting
jupyter notebook 02-sorting.ipynb
```

## **📈 Results**

Each notebook generates:

- CSV files with performance measurements
- Visualizations in various formats (PNG, PDF)
- Detailed analysis of algorithm behavior

## **📂 Project Structure**

```plaintext
📂 algorithms-analysis/
├── 📂 notebooks/               # Jupyter notebooks for analyses
│   ├── 📂 01-fibonacci/        # Fibonacci algorithm comparison
│   └── 📂 02-sorting/          # Sorting algorithms analysis
├── 📂 src/                     # Source code
│   ├── 🐍 performance.py       # Benchmarking utilities
│   ├── 🐍 plotting.py          # Data visualization tools
│   ├── 🐍 timing.py            # Time measurement functions
│   └── 🐍 utils.py             # Helper utilities
├── 📝 requirements.txt         # Package dependencies
├── 🔧 setup.py                 # Package installation configuration
├── 📄 LICENSE                  # MIT License
└── 📄 README.md                # This file
```

## **License**

The source code of this repository is licensed under the MIT License. See the [`LICENSE`](LICENSE) for more details.

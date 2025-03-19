"""
Sorting Algorithm Implementations
================================

This module provides different implementations of common sorting algorithms
with varying time and space complexity characteristics.

The module contains implementations of:
- Quick sort
- Merge sort
- Heap sort
- Insertion sort
"""


def quicksort(arr):
    """
    Sort an array using the quick sort algorithm.

    Uses the divide and conquer approach with a middle element as pivot.

    :param arr: The array to be sorted
    :type arr: list
    :return: A new sorted array
    :rtype: list

    :Time Complexity: Average O(n log n), Worst O(n^2)
    :Space Complexity: O(n) for the result and recursion stack
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]

    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def mergesort(arr):
    """
    Sort an array using the merge sort algorithm.

    Uses the divide and conquer approach by recursively splitting
    the array and merging the sorted subarrays.

    :param arr: The array to be sorted
    :type arr: list
    :return: A new sorted array
    :rtype: list

    :Time Complexity: O(n log n) in all cases
    :Space Complexity: O(n) for the result and recursion
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    """
    Merge two sorted arrays into a single sorted array.

    :param left: First sorted array
    :type left: list
    :param right: Second sorted array
    :type right: list
    :return: Merged sorted array
    :rtype: list

    :Time Complexity: O(n) where n is the total length of both arrays
    :Space Complexity: O(n) for the result array
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def heapify(arr, n, i):
    """
    Maintain the heap property for a binary heap.

    :param arr: The array representing a heap
    :type arr: list
    :param n: The size of the heap
    :type n: int
    :param i: The index of the current root
    :type i: int

    :Time Complexity: O(log n)
    :Space Complexity: O(1) in-place operation
    """
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heapsort(arr):
    """
    Sort an array in-place using the heap sort algorithm.

    :param arr: The array to be sorted
    :type arr: list

    :Time Complexity: O(n log n) in all cases
    :Space Complexity: O(1) in-place operation

    :Note: This function modifies the input array in-place
    """
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)


def insertionsort(arr):
    """
    Sort an array in-place using the insertion sort algorithm.

    :param arr: The array to be sorted
    :type arr: list

    :Time Complexity: Average and Worst O(n^2), Best O(n)
    :Space Complexity: O(1) in-place operation

    :Note: This function modifies the input array in-place
    :Note: Limited to arrays with length <= 1000 for efficiency
    """
    if len(arr) > 1000:
        return []
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
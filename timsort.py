def insertion_sort(arr, start, end):
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def timsort(arr):
    min_run = 32
    n = len(arr)

    # Insertion sort on each min_run-sized slice
    for i in range(0, n, min_run):
        insertion_sort(arr, i, min(i + min_run - 1, n - 1))

    # Merge adjacent min_run-sized slices
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(left + 2 * size - 1, n - 1)
            arr[left:right + 1] = merge(arr[left:mid + 1], arr[mid + 1:right + 1])
        size *= 2

    return arr
          
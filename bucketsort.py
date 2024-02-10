def bucket_sort(arr):
    max_num = max(arr)
    min_num = min(arr)
    num_buckets = (max_num - min_num) // 100 + 1
    buckets = [[] for _ in range(num_buckets)]

    for num in arr:
        index = (num - min_num) // 100
        buckets[index].append(num)

    for bucket in buckets:
        heap_sort(bucket)

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

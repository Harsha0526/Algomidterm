def quick_sort(array):
    if len(array) <= 1:
        return array

    pivot = array[len(array) // 2]
    less = []
    equal = []
    greater = []

    for x in array:
        if x < pivot:
            less.append(x)
        elif x == pivot:
            equal.append(x)
        elif x > pivot:
            greater.append(x)

    return quick_sort(less) + equal + quick_sort(greater)
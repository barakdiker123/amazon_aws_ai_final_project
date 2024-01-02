#!/usr/bin/env ipython


print("barak")
x = 5 + 3

print("moshea")


def test_add_function(x, y):
    """Test the add function"""
    return x + y


test_add_function(4, 5)


def search(n, a):
    low = 0
    high = len(a) - 1
    while True:
        mid = low + high // 2
        v = a[mid]
        if n == v:
            return mid
        if n < v:
            high = mid - 1
    return -1


i = search(1, [1, 2, 4])

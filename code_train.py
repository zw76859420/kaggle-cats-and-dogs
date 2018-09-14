#-*- coding:utf-8 -*-
#author:zhangwei

import numpy as np

def generator():
    i = 0
    while True:
        i += 1
        yield i
for k in generator():
    print(k)
    if k > 4:
        break
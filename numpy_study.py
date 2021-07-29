import numpy as np
import matplotlib.pyplot as plt

# arr1 = np.array([1, 2, 3])
# print(arr1)
# print(arr1[2])
#
# data = range(1, 10)
# arr = np.array(data)
# print(arr)
# print()
#
# arr2 = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr2)
# print(arr2[0, 2])
# print()
#
# arr3 = np.array([
#                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                 [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
#                 [[100, 200, 300], [400, 500, 600], [700, 800, 900]]
#                 ])
# print(arr3)
#
# zero_zero_zero = arr3[0, 0, 0]
# aaa = arr3[1, 2, 0]
#
# print(zero_zero_zero)
# print(aaa)
# print(arr3[0, 1, 0], arr3[1, 1, 1], arr3[2, 1, 2])
# print(arr3[0, 2, 2], arr3[1, 1, 1], arr3[2, 2, 0])

# arr1 = np.arange(64)
# print(arr1)
# print('=========================================================================')
#
# arr2 = arr1.reshape(4, 16)
# print(arr2)
# print('=========================================================================')
#
# arr3 = arr1.reshape(4, 4, 4)
# print(arr3)
# print('=========================================================================')

# zero1 = np.array([[0, 0], [0, 0]])
#
# zero2 = np.zeros((5, 5, 5))
# print(zero2)
#
# one1 = np.ones((3, 3, 3))
# print(one1, one1.dtype, sep='\n')
#
# unitMatrix = np.eye(5)
# print(unitMatrix)
#
# arr1 = np.arange(1, 30, 3)
# print(arr1)

# arr1 = np.linspace(0, 100, 5)
# print(arr1)

# arr1 = np.random.randint(10, 100, (2, 5))
# print(arr1)

# arr1 = np.zeros((3, 3))
# arr2 = np.ones((3, 3))
# data = np.arange(27)
# arr3 = data.reshape(3, 9)
# arr4 = data.reshape(3, 3, 3)
# arr5 = np.eye(3)  # 단위행렬 생성
# arr6 = np.arange(0, 30, 3)
#
# print(arr1)
# print(arr2)
# print(arr3)
# print(arr4)
# print(arr5)
# print(arr6)


# arr1 = np.random.randint(1, 100, (5, 5))
# print(arr1)
# print(arr1[2,2])
# print()
# arr2 = np.random.randint(1, 100, (3, 3, 3))
# print(arr2)
# print(arr2[1, 0, 1])
# print("최대값 :", arr2.max())
# print("최소값 :", arr2.min())
# print("평균값 :", arr2.mean())

# arr1 = np.full((2, 5), 30)
# print(arr1)
#
# arr1 = np.random.normal(10, 5, (3, 3))
# print(arr1)

# arr1 = np.random.randint(1, 11, (2, 5))
# print(arr1)
# print('=========================================================================')
# arr2 = np.random.randint(11, 20, (2, 5))
# print(arr2)
# print('=========================================================================')
# print("arr1+arr2 : ", arr1+arr2, sep='\n')
# print("arr1**2 : ", arr1**2, sep='\n')
# print("arr1/2 : ", arr1/2, sep='\n')
# print("(arr1+arr2)**2/2 : ", (arr1+arr2)**2/2, sep='\n')

# date1 = np.array('2021-07-26')
# print(date1, date1.dtype)
# print('=========================================================================')
# date2 = np.array('2021-07-26', dtype=np.datetime64)
# date3 = np.array('2021-09-12', dtype=np.datetime64)
# print(date3 - date2)
#
# arr1 = date2 + np.arange(30)
# print(arr1)
# print(arr1.shape)
# print(arr1.dtype)
# arr2 = arr1.reshape(3, 10)
# print(arr2.shape)
# print(arr2.dtype)
#
# arr1 = np.arange(date2, date3)
# print(arr1.shape)
# print(arr1.min(), arr1.max())
# arr2 = arr1.reshape(2, 4, 6)
# print(arr2)
# print(arr2.shape)
#
# h = np.datetime64('2021-07-26 12')
# m = np.datetime64('2021-07-26 12:34')
# s = np.datetime64('2021-07-26 12:34:12')
# print(h.dtype)
# print(m.dtype)
# print(s.dtype)
# s1 = np.datetime64('2021-07-26 12:34:12')
# s2 = np.datetime64('2021-07-27 12:34:12')
# print(s2-s1)
# a = s2-s1
# s1 = np.datetime64('2021-07-26 12:34')
# s2 = np.datetime64('2021-07-27 12:34')
# print(s2-s1)
# s1 = np.datetime64('2021-07-26 12')
# s2 = np.datetime64('2021-07-27 12')
# print(s2-s1)
# s1 = np.datetime64('2021-06-21 12:00:00')
# s2 = np.datetime64('2021-07-27 12:30:00')
# a = s2-s1
# print(a)


# arr1 = np.arange(1, 11)
# print(arr1.dtype)
# print("arr1.shape: ", arr1.shape)
# print(arr1)
# arr2 = arr1.reshape(2, 5)
# arr1_insert = np.insert(arr1, 0, 0)
# print(arr1_insert)
# print('=========================================================================')
# print("arr1_insert.shape: ", arr1_insert.shape)
# print(arr2)
# arr2_insert = np.insert(arr2, 2, range(10, 15), axis=0)
# print(arr2_insert)

# arr1 = np.arange(1, 28)
# arr2 = arr1.reshape(3, 3, 3)
# # print(arr2)
# arr2_insert = np.insert(arr2, 2, range(10, 13), axis=0)
# print(arr2_insert)

# arr1 = np.arange(1, 11)
# print(arr1)
#
# arr1[:6] = 100
# print(arr1)
#
# arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr2)
#
# arr2[0, :3] = 0
# print(arr2)

# arr1 = np.arange(1, 10)
# print(arr1)
#
# arr1_delete = np.delete(arr1, 0)
# print(arr1_delete)
#
# arr2 = arr1.reshape(3, 3)
# print(arr2)
# arr2_delete = np.delete(arr2, 2, axis=0)
# print(arr2_delete)


# arr1 = np.arange(1, 28)
# arr2 = arr1.reshape(3, 3, 3)
# print(arr2)
# arr2_delete0 = np.delete(arr2, 2, axis=0)
# arr2_delete1 = np.delete(arr2, 2, axis=1)
# arr2_delete2 = np.delete(arr2, 2, axis=2)
# print(arr2_delete0)
# print(arr2_delete1)
# print(arr2_delete2)

# arr1 = np.array(['1.5', '2', '2.7', '4'])
# print(arr1)
# print(arr1.dtype)

# change_arr1 = arr1.astype(int)
# print(change_arr1)
# print(change_arr1.dtype)

# change_arr1 = arr1.astype(float)
# print(change_arr1)
# print(change_arr1.dtype)
#
# change_int = change_arr1.astype(int)
# print(change_int)
# print(change_int.dtype)

# arr1 = np.arange(1, 11)
# print(arr1)
# print(arr1.sum())
# print(arr1[0] > 3)
# arr2 = np.arange(101, 111)
# print(arr1[0] < arr2[0])
# print(arr2)
# print(arr1+arr2)
# print(arr1 * 2)
# print(arr1 ** 2)
# print(arr1 << 2)
# print(arr1 >> 2)

# row = np.arange(2, 10)
# attr = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
# result = row * attr
# print(result.T)
# print(result.sum())
# print(result.mean())
# print(result.shape)
# print(result.T.shape)
# print(result.dtype)
# result_change = result.astype(float)
# print(result_change.dtype)

# arr = [(1, 2, 3), (4, 5, 6)]
# print(arr)
# a = np.array(arr, dtype=float)
# print(a.dtype, a, sep='\n')

# a = np.linspace(0, 1, 5)
# print(a)
# plt.plot(a, '^')
# # plt.show()
# # plt.hist()
# np.save("/datas", a)

# arr = np.arange(64)
# arr1 = arr.reshape(4, 16)
# arr2 = arr.reshape(4, 4, 4)
# print(arr.ndim)
# print(arr1.ndim)
# print(arr2.ndim)

a1 = np.datetime64('2021-07-26 16:40:00')
a2 = np.datetime64('2021-09-21 00:00:00')
print(a2-a1)
a1 = np.datetime64('2021-07-26 16:40')
a2 = np.datetime64('2021-09-21 00:00')
print(a2-a1)
a1 = np.datetime64('2021-07-26 16')
a2 = np.datetime64('2021-09-21 00')
print(a2-a1)
a1 = np.datetime64('2021-07-26')
a2 = np.datetime64('2021-09-21')
print(a2-a1)





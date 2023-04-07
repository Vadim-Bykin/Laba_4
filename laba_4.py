# Формируется матрица F следующим образом: скопировать в нее А и если в С количество нулевых элементов в нечетных столбцах больше, чем количество нулевых элементов в четных столбцах,
# то поменять местами С и В симметрично, иначе С и Е поменять местами несимметрично. При этом матрица А не меняется.
# После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то вычисляется выражение: A * AT – K * FТ,
# иначе вычисляется выражение (AТ + G - F-1) * K, где G - нижняя треугольная матрица, полученная из А.
# Выводятся по мере формирования А, F и все матричные операции последовательно.
# E B
# D C

import numpy as np
import matplotlib.pyplot as plt

K_test = 3
N_test = 11
A_test = np.asarray((
    [5, -1, 3, 1, -9, 4, -9, 8, -1, 5, 3],
    [8, -4, -8, 1, -5, 2, -4, -8, -5, -1, 3],
    [-6, 5, -2, -10, 2, 8, -3, 3, 5, -1, -9],
    [-6, -1, 10, 4, -1, -5, 2, 7, 3, 8, 0],
    [3, 6, -4, 4, -2, -4, 3, 3, -2, 4, -1],
    [-1, 8, 6, 10, -3, 7, -8, -9, -9, -5, 2],
    [-2, -6, -3, 1, 2, -3, -3, 5, -7, 1, 7],
    [-7, 10, -4, -8, -9, -9, 4, 5, 7, -10, -4],
    [3, 6, -9, 7, 2, -10, 4, -9, 2, -4, 0],
    [10, 9, 9, 5, 8, -8, -8, -1, 10, -7, 1],
    [6, -5, 0, 1, -5, -3, -5, 4, 7, -2, 0]))

print('Использовать тестовые данные или случайные?')
choice = input('Ваш выбор (1 - тестовые данные, 2 - случайные, q-выход): ')

if choice == '1':
    K = K_test
    N = N_test
    A = A_test

if choice == '2':
    K = int(input('Введите K: '))
    N = int(input('Введите N: '))

    if (N < 2):
        print('Ошибка в исходных данных. Длина сторон матрицы А (N,N) должна быть больше 1!')
        exit()

    A = np.random.randint(low=-10, high=11, size=(N, N))  # [low, high)

n = N // 2  # размерность матриц B, C, D, E (n x n)

w = N // 2
if N % 2 == 0:
    E = A[0:w, 0:w]
    B = A[0:w, w:]
    C = A[w:, w:]
    D = A[w:, 0:w]
else:
    E = A[0:w, 0:w]
    B = A[0:w, w + 1:]
    C = A[w + 1:, w + 1:]
    D = A[w + 1:, 0:w]

if choice == 'q':
    exit()

# печатаем матрицы E, B, C, D, A
print('Матрица E:')
print(E)

print('Матрица B:')
print(B)

print('Матрица C:')
print(C)

print('Матрица D:')
print(D)

print('Матрица A:')
print(A)

count_zero_c_even = 0  # количество нулевых элементов в четных столбцах
count_zero_c_odd = 0  # количество нулевых элементов в нечетных столбцах

for col in range(0, n, 2):  # четные столбцы
    for row in range(n):
        if C[row][col] == 0:
            count_zero_c_even += 1

for col in range(1, n, 2):  # нечетные столбцы
    for row in range(n):
        if C[row][col] == 0:
            count_zero_c_odd += 1

F = A.copy()
if count_zero_c_odd > count_zero_c_even:
    print('Меняем местами B и C симметрично')
    if N % 2 == 0:
        F[0:w, w:] = np.flipud(C)  # flipud - отражение по вертикали,
        F[w:, w:] = np.flipud(B)  # fliplr - по горизонтали, flip - относительно вертикали и горизонтали
    else:
        F[0:w, w + 1:] = np.flipud(C)
        F[w + 1:, w + 1:] = np.flipud(B)
else:
    print('Меняем местами E и C несимметрично')
    if N % 2 == 0:
        F[0:w, 0:w] = C
        F[w:, w:] = E
    else:
        F[0:w, 0:w] = C
        F[w + 1:, w + 1:] = E

print('Матрица F')
print(F)

det_A = np.linalg.det(A)  # определитель матрицы A
sum_diag = np.trace(F)  # сумма диагональных элементов матрицы F

if det_A > sum_diag:  # определитель матрицы A больше суммы диагональных элементов матрицы F?
    print('определитель матрицы A больше суммы диагональных элементов матрицы F')
    result = A.dot(A.T) - K * F.T
else:
    if np.linalg.det(F) == 0:
        print('Определитель матрицы F равен 0 => обратная матрица для F не существует')
        result = 0
    else:
        G = np.tril(A, -1)  # нижняя трегугольная матрица из матрицы A
        result = (A.T + G - np.linalg.inv(F)) * K

np.set_printoptions(precision=1, suppress=True)  # выводим с точностью до одного знака после запятой и без e
print('Результат:')
print(result)

#Работа с графиками
plt.figure(figsize=(16, 9))

# вывод тепловой карты матрицы F
plt.subplot(2, 2, 1)
plt.xticks(ticks=np.arange(F.shape[1]))
plt.yticks(ticks=np.arange(F.shape[1]))
plt.xlabel('Номер столбца')
plt.ylabel('Номер строки')
hm = plt.imshow(F, cmap='Oranges', interpolation="nearest")
plt.colorbar(hm)
plt.title('Тепловая карта элементов')

# вывод диаграммы распределения сумм элементов по строкам в матрице F
sum_by_rows = np.sum(F, axis=1)  # axis = 1 - сумма по строкам
x = np.arange(F.shape[1])
plt.subplot(2, 2, 2)
plt.plot(x, sum_by_rows, label='Сумма элементов по строкам')
plt.xlabel('Номер строки')
plt.ylabel('Сумма элементов')
plt.title('График суммы элементов по строкам')
plt.legend()

# вывод диаграммы распределения количества положительных элементов в столбцах матрицы F
res = []
for col in F.T:
    count = 0
    for el in col:
        if el > 0:
            count += 1
    res.append(count)

x = np.arange(F.shape[1])
plt.subplot(2, 2, 3)
plt.bar(x, res, label='Количество положительных элементов в столбцах')
plt.xlabel('Номер столбца')
plt.ylabel('Количество положительных элементов')
plt.title('График количества положительных элементов в столбцах')
plt.legend()

# вывод круговой диаграммы
x = np.arange(F.shape[1])
plt.subplot(2, 2, 4)
P = []
for i in range(N):
    P.append(abs(F[0][i]))
plt.pie(P, labels=x, autopct='%1.2f%%')
plt.title("График с использованием функции pie")

plt.tight_layout(pad=3.5, w_pad=3, h_pad=4) # расстояние от границ и между областями
plt.suptitle("Использование библиотеки Matplotlib", y=1)
plt.show()

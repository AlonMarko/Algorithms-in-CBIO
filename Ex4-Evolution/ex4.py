import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm

NUCLEOTIDES = ["A", "C", "G", "T"]
ALPHA = 1
T = [0.15, 0.4, 0.9]
np.seterr(invalid='ignore', divide='ignore')


def procedure(t, n, a="A"):
    nuc_list = [a] + [b for b in NUCLEOTIDES if b != a]
    p_a_eq_b = (1 + 3 * pow(math.e, -4 * ALPHA * t)) / 4
    p_a_neq_b = (1 - pow(math.e, -4 * ALPHA * t)) / 4
    return random.choices(nuc_list, weights=[p_a_eq_b, p_a_neq_b, p_a_neq_b, p_a_neq_b], k=n), p_a_eq_b, p_a_neq_b


def Q1_2b():
    """
    we chose a=A, b=C
    :return:
    """
    N = [10, 100, 1000, 100000]
    df_actual = pd.DataFrame(index=N, columns=T)
    df_prediction = pd.DataFrame(index=N, columns=T)
    df_actual_neq = pd.DataFrame(index=N, columns=T)
    df_prediction_neq = pd.DataFrame(index=N, columns=T)
    for t in T:
        for n in N:
            results, aeq, neq = procedure(t, n)
            df_actual[t][n] = results.count("A") / n
            df_prediction[t][n] = aeq
            df_prediction_neq[t][n] = neq
            df_actual_neq[t][n] = results.count("C") / n

    print("actual, a=b:")
    print(df_actual)
    print("\n\nprediction, a=b:")
    print(df_prediction)
    print("\n\nactual, a!=b:")
    print(df_actual_neq)
    print("\n\nprediction, a!=b:")
    print(df_prediction_neq)


def Q1_3a(t):
    a = random.choice(NUCLEOTIDES)
    b, _, _ = procedure(t, 1, a)
    return a, b[0]


def Q1_3b():
    N = 500
    M = 100
    MLES = []
    for t in T:
        MLE = []
        for _ in range(M):
            count_eq = 0
            count_neq = 0
            for _ in range(N):
                a, b = Q1_3a(t)
                if a == b:
                    count_eq += 1
                else:
                    count_neq += 1
            mle = np.log((3 * count_eq - count_neq) / (3 * count_eq + 3 * count_neq)) / (-4 * ALPHA)
            # normalize results
            if not np.isinf(mle) and not np.isnan(mle):
                MLE.append(mle)

        print(f'\nMedian of t={t} is {round(np.median(MLE), 2)}')
        MLES.append(MLE)
    df = pd.DataFrame(MLES, index=T).T
    _, ax = plt.subplots()
    ax = df.boxplot(column=T)
    ax.set_xlabel("t")
    ax.set_ylabel("MLE")
    ax.set_title("MLE against t")
    plt.show()


def Q2_helper(R):
    t_s = [1, 2, 3, 4, 5, 6, 7]
    for t in t_s:
        print(f't={t}')
        print(expm(t * R))


def Q2():
    print("\n\nQ2-a")
    Q2_helper(np.array([[-4, 2, 1, 1], [1, -4, 2, 1], [1, 1, -4, 2], [2, 1, 1, -4]]))
    print("\n\nQ2-b")
    Q2_helper(np.array([[-3, 1, 1, 1], [2, -4, 1, 1], [2, 1, -4, 1], [2, 1, 1, -4]]))


if __name__ == '__main__':
    Q1_2b()
    Q1_3b()
    Q2()

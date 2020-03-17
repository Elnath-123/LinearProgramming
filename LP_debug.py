import numpy as np
import re
import os
import sys
import time

def normalization(func_type, v_c, mat_constraint, v_condition):
    # objective function must be max
    print(func_type)
    transform = False
    if func_type == 'min':
        transform = True
        v_c = -1 * v_c

    # resources vector must be positive
    for i, constraint in enumerate(mat_constraint):
        # print(type(mat_constraint))
        if constraint[-1] < 0:
            mat_constraint[i, ...] = -1 * mat_constraint[i, ...]
            if v_condition[i] == '<=':
                v_condition[i] = '>='
            elif v_condition[i] == '>=':
                v_condition[i] = '<='

    # transform the inequality to equality according to v_condition
    assist_mat = []
    artifact = []
    n = len(mat_constraint[0])
    for i, condition in enumerate(v_condition):
        assist_mat.append([])
        if i > 0:
            for j in range(len(assist_mat[i - 1])):
                assist_mat[i].append(0)
        if condition == '<=':
            assist_mat[i].append(1)
            n = n + 1
        elif condition == '=':
            assist_mat[i].append(1)
            artifact.append((i, n - 1))
            n = n + 1
        elif condition == '>=':
            assist_mat[i].append(-1)
            n = n + 1
            assist_mat[i].append(1)
            artifact.append((i, n - 1))
            n = n + 1


    print(artifact)
    maxlen = len(max([elem for elem in assist_mat], key=len))
    for i in range(len(v_condition)):
        while len(assist_mat[i]) < maxlen:
            assist_mat[i].append(0)
    b = mat_constraint[:, -1].reshape(1, -1).transpose()
    mat_constraint = np.delete(mat_constraint, -1, axis=1)
    mat_constraint = np.concatenate((mat_constraint,assist_mat), axis=-1)
    mat_constraint = np.concatenate((mat_constraint, b), axis=-1)

    while len(v_c) < len(mat_constraint[0]) - 1:
        v_c = np.concatenate((v_c, np.array([0])))
    return v_c, mat_constraint, artifact, transform


def lp_input():
    print('''Please input your objective function's value vector  
                 format: value vector c = (c1, c2, ..., cn)T  
              ''')
    objective_function = input()
    func_type = re.match("(\w+)", objective_function).group()
    print(func_type)
    v_c = list(map(float, np.array(re.findall(r"-?\d+\.?\d*", objective_function))))
    print('Please input number of constraints A')
    constraint_num = int(input())
    mat_constraint = []
    v_condition = []
    for i in range(constraint_num):
        print('''Please input the next constraint
        format: (ai1, ai2, ..., ain) (<= or >= or =) bi 
        ''')
        str_constraint = input()
        v_constraint = list(map(float, np.array(re.findall(r"-?\d+\.?\d*", str_constraint))))
        mat_constraint.append(v_constraint)
        v_condition.append(re.findall(r"<=|>=|=", str_constraint))
    v_condition = np.array(v_condition).flatten()
    v_c, mat_constraint, artifact, transform = normalization(func_type, np.array(v_c), np.array(mat_constraint), v_condition)
    return v_c, mat_constraint, artifact, transform


def find_base_index(v_base, mat_constraint):
    print(v_base)
    for i in range(len(mat_constraint[0])):
        if (mat_constraint[:, i] == v_base).all():
            return i


def simplex_method(v_c, mat_constraint, artifact, transform):
    inf = 800
    M = -800
    for tup in artifact:
        v_c[tup[1]] = M
    print(v_c)
    cj_zj = v_c.copy()
    m = len(mat_constraint)
    n = len(mat_constraint[0])
    x = np.zeros(len(v_c))
    cb = [0] * m
    base = [0] * m
    non_base = [0] * (n - m)
    iter = 0
    print("v_c = " + str(v_c))
    while True:


        base.clear()
        non_base.clear()
        for i in range(m):
            base.append(find_base_index(np.array([(j == i) for j in range(m)]) + 0, mat_constraint))
        for i in range(len(cj_zj)):
            if i not in base:
                non_base.append(i)
        print("non_base = " + str(non_base))
        print("base = " + str(base))

        v_b = mat_constraint[:, -1]
        for i in range(m):
            cb[i] = v_c[base[i]]

        for i in range(n - 1):
            cj_zj[i] = v_c[i] - np.inner(cb, mat_constraint[:, i])

        if max(cj_zj) <= 0:
            break

        for i in range(m, n - 1):
            if cj_zj[i] > 0 and (mat_constraint[:, i] < np.array([0 for j in range(m)])).all():
                print("[Hint] Your lp problem has Unbounded Solution!")
                sys.exit(0)

        print("v_c = " + str(v_c))
        print("cj_zj = " + str(cj_zj))
        target_in_idx = np.argmax(cj_zj)
        v_target_c = mat_constraint[:, target_in_idx]
        print("v_target_c = " + str(mat_constraint[:, target_in_idx]))
        theta = np.array(np.zeros(len(v_b)))
        for i in range(len(v_b)):
            if v_target_c[i] == 0 or v_target_c[i] < 0:
                theta[i] = inf
            else:
                theta[i] = v_b[i] / v_target_c[i]
        target_out_idx = np.argmin(theta)
        print("theta = " + str(theta))
        pivot = mat_constraint[target_out_idx, target_in_idx]
        print("pivot = " + str(pivot))
        mat_constraint[target_out_idx, :] = mat_constraint[target_out_idx, :] / pivot
        for i in range(m):
            if i != target_out_idx and mat_constraint[i, target_in_idx] != 0:
                px = mat_constraint[i, target_in_idx]
                v_after_px = mat_constraint[target_out_idx, :] * px
                mat_constraint[i, :] -= v_after_px

        print("cb =" + str(cb))
        print(mat_constraint)
        iter = iter + 1

    b = mat_constraint[:, -1]
    for i in range(len(b)):
        x[base[i]] = b[i]
    print(x)
    best_solution = np.inner(x, v_c)

    if transform == True:
        best_solution = -1 * best_solution

    for artifact_var in artifact:
        if artifact_var[1] in base:
            print("[Hint] Your artificial variable in the base, No solution!")
            sys.exit(0)

    print(best_solution)
    return x, best_solution, iter

def solve_lp():
    start_time = time.time()
    v_c, mat_constraint, artifact, transform = lp_input()
    print(mat_constraint)
    x, best_solution, iter_time = simplex_method(v_c, mat_constraint, artifact, transform)
    end_time = time.time()

    print("Final Solution: ")
    print("X* = " + str(x))
    print("Object value = " + str(np.array(best_solution).transpose()))
    print("----------------------------------")
    print("Times of iteration : " + str(iter_time))
    print("Solved in %f sec" % (end_time - start_time))

if __name__ == '__main__':
    solve_lp()
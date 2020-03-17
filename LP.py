import numpy as np
import re
import os
import sys
import time
class Simplex:
    def normalization(self, func_type, v_c, mat_constraint, v_condition):

        mat_constraint = np.array(mat_constraint)
        v_c = np.array(v_c)
        v_condition = np.array(v_condition).flatten()

        n = len(mat_constraint[0])
        m = len(mat_constraint)

        # objective function must be max
        transform = False
        if func_type == 'min':
            transform = True
            v_c = -1 * v_c
        # resources vector must be positive
        for i, constraint in enumerate(mat_constraint):
            if constraint[-1] < 0:
                mat_constraint[i, ...] = -1 * mat_constraint[i, ...]
                if v_condition[i] == '<=':
                    v_condition[i] = '>='
                elif v_condition[i] == '>=':
                    v_condition[i] = '<='

        # transform the inequality to equality according to v_condition
        assist_mat = []
        artifact = []
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

        # padding artificial variables with zeros
        maxlen = len(max([elem for elem in assist_mat], key=len))
        for i in range(m):
            while len(assist_mat[i]) < maxlen:
                assist_mat[i].append(0)

        # add artificial variables into mat_constraint
        b = mat_constraint[:, -1].reshape(1, -1).transpose()
        mat_constraint = np.delete(mat_constraint, -1, axis=1)
        mat_constraint = np.concatenate((mat_constraint,assist_mat), axis=-1)
        mat_constraint = np.concatenate((mat_constraint, b), axis=-1)

        # pad resources vector v_c with zeros
        while len(v_c) < len(mat_constraint[0]) - 1:
            v_c = np.concatenate((v_c, np.array([0])))
        return v_c, mat_constraint, artifact, transform


    def lp_input(self):
        print('''Please input your objective function's value vector  
                     format: value vector c = (c1, c2, ..., cn)T  
                  ''')
        objective_function = input()
        # read the function type (max or min)
        func_type = re.match("(\w+)", objective_function).group()

        # read the resources vector v_c
        v_c = list(map(float, np.array(re.findall(r"-?\d+\.?\d*", objective_function))))

        # read the constraint as mat_constraint
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
        # v_condition = np.array(v_condition).flatten()


        # normalize the problem
        v_c, mat_constraint, artifact, transform = self.normalization(func_type, v_c, mat_constraint, v_condition)
        return v_c, mat_constraint, artifact, transform

    # find the identity matrix in mat_constraint
    # return: every column number of the identity
    def find_base_index(self, v_base, mat_constraint):
        for i in range(len(mat_constraint[0])):
            if (mat_constraint[:, i] == v_base).all():
                return i

    # simplex method to solve the linear
    # programming problem with big M method
    # theta with no actual meaning will be treated as inf(8000)
    # return: x*, value of solution, times of iteration
    def simplex_method(self, v_c, mat_constraint, artifact, transform):
        # record time
        start_time = time.time()

        inf = 8000
        # M value
        M = -8000

        m = len(mat_constraint)
        n = len(mat_constraint[0])

        # get the position of artifacial value
        for tup in artifact:
            v_c[tup[1]] = M

        # relative vector used in iteration processes
        cj_zj = [0] * len(v_c)
        cb = [0] * m
        base = [0] * m  # variables in base
        non_base = [0] * (n - m)  # variables do not in base

        # best solution X*
        x = np.zeros(len(v_c))

        # simulate the process of simplex method
        iter = 0
        while True:
            base.clear()
            non_base.clear()

            # find current varibles in base
            for i in range(m):
                base.append(self.find_base_index(np.array([(j == i) for j in range(m)]) + 0, mat_constraint))

            # find current varibles do not in base
            for i in range(len(cj_zj)):
                if i not in base:
                    non_base.append(i)

            # take the v_b out from constraint matrix
            v_b = mat_constraint[:, -1]
            for i in range(m):
                cb[i] = v_c[base[i]]

            # calculate cj-zj vector
            for i in range(n - 1):
                cj_zj[i] = v_c[i] - np.inner(cb, mat_constraint[:, i])

            # check if the iteration should be ended
            if max(cj_zj) <= 0:
                break

            # check if the solution is unbounded
            for i in range(m, n - 1):
                if cj_zj[i] > 0 and (mat_constraint[:, i] < np.array([0 for j in range(m)])).all():
                    print("[Hint] lp problem has UNBOUNDED SOLUTION!")
                    sys.exit(0)

            # find target varibles should be switched in base
            target_in_idx = np.argmax(cj_zj)
            v_target_c = mat_constraint[:, target_in_idx]

            # calculate theta
            theta = np.array(np.zeros(len(v_b)))
            for i in range(len(v_b)):
                if v_target_c[i] == 0 or v_target_c[i] < 0:
                    theta[i] = inf
                else:
                    theta[i] = v_b[i] / v_target_c[i]

            # find tartget variables should be swiched out of base
            target_out_idx = np.argmin(theta)

            # take gaussian elimination
            pivot = mat_constraint[target_out_idx, target_in_idx]
            mat_constraint[target_out_idx, :] = mat_constraint[target_out_idx, :] / pivot
            for i in range(m):
                if i != target_out_idx and mat_constraint[i, target_in_idx] != 0:
                    px = mat_constraint[i, target_in_idx]
                    v_after_px = mat_constraint[target_out_idx, :] * px
                    mat_constraint[i, :] -= v_after_px

            iter = iter + 1

        # calculate the final solution
        b = mat_constraint[:, -1]
        for i in range(len(b)):
            x[base[i]] = b[i]
        best_solution = np.inner(x, v_c)

        if transform == True:
            best_solution = -1 * best_solution

        # check if the existence of solution
        for artifact_var in artifact:
            if artifact_var[1] in base:
                print("[Hint] Your artificial variable in the base, NO SOLUTION!")
                sys.exit(0)

        end_time = time.time()
        times = end_time - start_time

        return x, best_solution, iter, times

    def print_solution(self, x, best_solution, iter_times, exe_time):
        print("Final Solution: ")
        print("X* = " + str(np.array(x).transpose()))
        print("Object value = " + str(best_solution))
        print("----------------------------------")
        print("Times of iteration : " + str(iter_times))
        print("Solved in %f sec" % exe_time)

    def solve_lp_manual_input(self, func_type, v_c, mat_constraint, v_condition):
        v_c, mat_constraint, artifact, transform = self.normalization(func_type, v_c, mat_constraint, v_condition)
        print(mat_constratint)
        x, best_solution, iter_time, times = self.simplex_method(v_c, mat_constraint, artifact, transform)
        self.print_solution(x, best_solution, iter_time, times)

    def solve_lp(self):
        v_c, mat_constraint, artifact, transform = self.lp_input()
        x, best_solution, iter_time , times = self.simplex_method(v_c, mat_constraint, artifact, transform)

        self.print_solution(x, best_solution, iter_time, times)


if __name__ == '__main__':
    s = Simplex()
    mat_constratint = np.array([[1.0, 2.0, 8.0], [4.0, 0.0, 16.0], [0.0, 4.0, 12.0]])
    s.solve_lp_manual_input('max', [2.0, 3.0], mat_constratint, ['<=', '<=', '<='])
    # s.solve_lp()
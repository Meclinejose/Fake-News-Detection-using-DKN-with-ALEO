import random,math
import numpy as np



def Algm():
    def generate(n, m, l, u):   #Initial position
        data = []
        for i in range(n):
            tem = []
            for j in range(m):
                tem.append(random.uniform(l, u))
            data.append(tem)
        return data

    N, M, lb, ub = 10, 5, 1, 5
    g, max_itr = 0, 100

    soln = generate(N, M, lb, ub)


    def fitness(soln):
        F = []
        for i in range(len(soln)):
            F.append(random.random())
        return F
    gBest, gfit = [], float('inf')

    def Exploration_operation(S_vector):
        r1_vector=random.randint(1,2)
        r2_vector=random.randint(1,2)
        D_Vector=[]
        for i in range(len(S_vector)):
            temp=[]
            for j in range(len(S_vector[i])):
                temp.append(r1_vector*(S_vector[i][j]-1))
            D_Vector.append(temp)

        S_vector_=[]
        for i in range(len(D_Vector)):
            temp=[]
            for j in range(len(D_Vector[i])):
                temp.append(S_vector[i][j]+D_Vector[i][j]*(2*r2_vector-1))
            S_vector_.append(temp)

        return S_vector_


    def Explotaition_operation(S_vector,Lt):
        r2_vector = random.randint(1,2)
        r3_vector = random.randint(1,2)

        S_vector_=[]
        for i in range(len(S_vector)):
            temp=[]
            for j in range(len(S_vector[i])):
                D_vector=r3_vector*(Lt[j]-S_vector[i][j])
                temp.append(r2_vector*(S_vector[i][j]+D_vector))
            S_vector_.append(temp)
        return S_vector_

    def mutation_operation(S_vector):
        r=random.randint(1,2)
        z=random.uniform(0,1)
        N=max_itr
        h,x=10,10
        S_vector_=[]

        for i in range(len(S_vector)):
            temp=[]
            for j in range(len(S_vector[i])):
                t=j+1
                k_vector=z+((2*np.square(t)/np.square(N)))
                temp.append(r*(S_vector[i][j]+k_vector))
            S_vector_.append(temp)

        upd_soln=[]
        for i in range(len(S_vector)):
            temp=[]
            for j in range(len(S_vector[i])):
                t = j + 1
                k_vector = z + ((2 * np.square(t) / np.square(N)))
                temp.append(k_vector*np.square(z)-h*(np.cos(x)/(1-np.cos(x))))
            upd_soln.append(temp)
        return upd_soln




    while g<max_itr:
        fit = fitness(soln)
        bst = np.argmax(fit)
        S_vect=Exploration_operation(soln)
        Lt = S_vect[bst]
        S_vect_=Explotaition_operation(S_vect,Lt)
        upd_s_vect=mutation_operation(S_vect_)
        bst_soln=upd_s_vect[bst]

        g+=1

    return np.max(bst_soln)
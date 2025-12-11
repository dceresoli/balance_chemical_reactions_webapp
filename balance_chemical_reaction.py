#!/usr/bin/env python

# balance chemical reactions using linear algebra
# Davide Ceresoli <dceresoli@gmail.com>
# Programming for Chemistry 2025-2026 @ UniMI

import numpy as np
import scipy
import math
from parse_chemical_formula import formula_to_dict

# Given a chemical equation in a string like "O2 + H2 = H2O", write a function
# that splits the string and returns a list of reactants and a list of products
def get_reactants_prodcuts(eq):
    lhs, rhs = eq.split('=')
    reactants = [x.strip() for x in lhs.split('+')]
    products = [x.strip() for x in rhs.split('+')]
    return reactants, products


# Write a function that build the A matrix, given the list of reactants
# and products. Use formula_to_dict to extract the chemical formula
def build_matrix(reactants, products, debug=False):
    # number of molecules
    nmols = len(reactants) + len(products)
    
    # count number of elements
    elements = []
    for m in reactants + products:
        for f in formula_to_dict(m).keys():
            if f not in elements:
                elements.append(f)
    
    nelements = len(elements)

    if debug:
        print('molecules =', reactants+products)
        print('elements =', elements)

    A = np.zeros((nelements,nmols), dtype=np.float64)

    # fill the matrix: for each molecule (colunm), fill the corresponding
    # row (element) with the proper coefficient
    for i in range(len(reactants)):
        mydict = formula_to_dict(reactants[i])
        for j in range(nelements):
            if elements[j] in mydict:
                A[j,i] = mydict[elements[j]]

    # do the same for the products but with opposite sign
    for i in range(len(products)):
        mydict = formula_to_dict(products[i])
        for j in range(nelements):
            if elements[j] in mydict:
                A[j,i+len(reactants)] = -mydict[elements[j]]

    return A


# Write a function to solve the homogenenous linear system.
# 1. To get the the null space, use 'scipy.linalg.null_space(A)'.
# 2. If the null space has zero length, there is no solution.
#    If the length is >1, there are more solutions and let's think more about it.
# 3. If the length is 1, find the least common multiple that makes it a vector of integers.
# 4. Check that the integer vector is the correct solution)
def solve_linear_system(A):
    # get the null space
    x = scipy.linalg.null_space(A)
    
    # check
    if x.shape[1] == 0:
        raise RuntimeError("the reaction cannot be balanced!")
    elif x.shape[1] > 1:
        raise RuntimeError("the reaction can be balanced in multiple ways!")
    
    # extract the vector, divide by the smallest value
    c = x[:,0]
    c = c / np.min(c)

    # find the l.c.m.
    for i in range(1,1000):
        cint = c*i
        if np.all( np.abs(np.round(cint) - cint) < 1e-9 ):
            break
    
    # check if the solution is correct
    zero = A @ cint
    if np.any(np.abs(zero) > 1e-9):
        raise RuntimeError("could not find integer coefficients")

    return cint


# write a function that calls get_reactants_products(), build_matrix() and
# solve_linear_system(), then returns the balanced reaction as a string
# like: "O2 + 2*H2 = 2*H2O"
def balance_chemical_reaction(eq):
    reactants, products = get_reactants_prodcuts(eq)
    A = build_matrix(reactants, products, debug=False)
    c = solve_linear_system(A)

    lhs = []
    for i in range(len(reactants)):
        coeff = int(round(c[i]))
        if coeff == 1:
            lhs.append(reactants[i])
        else:
            lhs.append(str(coeff)+'·'+reactants[i])

    rhs = []
    for i in range(len(products)):
        coeff = int(round(c[i+len(reactants)]))
        if coeff == 1:
            rhs.append(products[i])
        else:
            rhs.append(str(coeff)+'·'+products[i])

    return ' + '.join(lhs) + ' = ' + ' + '.join(rhs)        


if __name__ == '__main__':
    equations = ['O2 + H2 = H2O', 'CH4+O2 =CO2+ H2O', 'MnS + As2Cr10O35 + H2SO4 = HMnO4 + AsH3 + CrS3O12 + H2O']
    for eq in equations:
        balanced = balance_chemical_reaction(eq)
        print(eq, '  =>  ', balanced)


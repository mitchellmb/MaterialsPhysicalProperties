#Generation of matrices used in crystalline electric field calculations determining single-ion eigenenergies

import numpy as np

def J_matrices(J_value):
    #create a diagonal matrix of J to -J
    J_z = np.matrix(np.diag(np.arange(-J_value,J_value+1.,1.)[::-1])) 

    #Identity matrix
    I = []
    for i in range(0,int(2.*J_value+1)):
        I.append(1.)
    I_J = np.diag(I)

    # X matrix
    Jd = []
    for i in range(0,int(2.*J_value+1)):
        Jd.append(J_value)
    J_J = np.diag(Jd)
    X_J = J_J*(J_J+(1.*I_J))

    #raising goes from +1 up to length(matrix_J)
    def raising(J,mJ, level):
        return np.sqrt(J*(J+1.)-(mJ+level)*(mJ+level+1.))  
      
    #lowering goes from -1 up to length(matrix_J-2)
    def lowering(J,mJ, level):
        return np.sqrt(J*(J+1.)-(mJ+level)*(mJ+level-1.))
        
    raising_components = []
    lowering_components = []
    for i in range(0,int(2.*J_value+1)):
        raising_components.append(raising(J_value,-J_value,i))
        lowering_components.append(lowering(J_value,-J_value,i))

    raisers = raising_components[::-1] #skip the first value since you cannot raise highest mJ higher
    lowerers = lowering_components[::-1] #skip the last value since you cannot lower past lowest mJ

    J_plus_J = np.matrix(np.zeros(np.shape(J_z)))
    J_minus_J = np.matrix(np.zeros(np.shape(J_z)))

    for i in range(0,len(raising_components)-1): #skips first raising mJ and last lowering mJ since they're zero and cannot raise/lower past the largest/lowest value
        J_plus_J[i,i+1] = raisers[i+1]
        J_minus_J[i+1,i] = lowerers[i]
        
    return J_z,J_plus_J,J_minus_J,X_J

def probability_of_transition(vec1, vec2, J_zz, J_plus_Jj, J_minus_Jj): 
    '''
    Calculate the dipolar transition probabilities with created matrices 
    vec1 must be ground state, vec2 is the exicted state
    calculating sum(<Excited|{Jz,J+,J-}|Ground>^2) 
    '''
    Jz_prob = np.abs(np.matmul(vec2,np.transpose(np.matmul(J_zz,vec1)))) #Jz
    Jx_prob = np.abs(np.matmul(vec2,np.transpose(np.matmul((1./2.*(J_plus_Jj+J_minus_Jj)),vec1)))) 
    Jy_prob = np.abs(np.matmul(vec2,np.transpose(np.matmul((1./2.*(J_minus_Jj-J_plus_Jj)),vec1)))) 
    return (a**2+b**2+c**2)

def quadrupolar_operators(Jz, Jp, Jm, XJ):
    '''
    Create quadrupolar matrice
    '''
    Jx = 1./2.*(Jp+Jm)
    Jy = 1./(2.j)*(Jp-Jm)
    
    #gamma3+ quad moments
    op1 = 3.*Jz**2-XJ
    op2 = Jx**2-Jy**2
    
    #gamma5+ quad moments
    op3 = 1./2.*(np.matmul(Jx,Jy)+np.matmul(Jy,Jx))
    op4 = 1./2.*(np.matmul(Jx,Jz)+np.matmul(Jz,Jx)) 
    op5 = 1./2.*(np.matmul(Jy,Jz)+np.matmul(Jz,Jy))
    
    return [op1, op2, op3, op4, op5]

def probability_of_transition_quad(vec1, vec2, quad_operators): 
    '''
    Calculate the quadrupolar transition probabilities with created matrices
    '''
    probabilities = [np.power(np.abs(np.matmul(vec2,np.transpose(np.matmul(quad_operators[i],vec1)))), 2.)
                     for i in range(0, len(quad_operators))]
    return sum(probabilities)

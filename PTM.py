import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




sigma = np.array([[[1, 0], [0, 1]],
                  [[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype = complex)




T_matrix = np.array([[1, 0, 0, 1],
                          [0, 1, -1j, 0],
                          [0, 1, 1j, 0],
                          [1, 0, 0, -1]]) # transformation matrix from odinary basis to Pauli basis
                                          # vec(in ordinary) = trans_matrix * vec (in Pauli)
T_matrix_inv = np.linalg.inv(T_matrix)
T_matrix2 = np.kron(T_matrix, T_matrix)
T_matrix_inv2 = np.kron(T_matrix_inv, T_matrix_inv)
rhovstack_to_rhovec = np.array([[ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]])
T_process_to_ptm = rhovstack_to_rhovec@T_matrix2
T_process_to_ptminv = T_matrix_inv2@rhovstack_to_rhovec




def propagator_to_ptm(propagator):
    '''
    Convert process matrix to PTM.

    Params
    ------
        propagator : process matrix

    Returns
    -------
        (ndarray)  : PTM
    '''
    return T_process_to_ptminv@propagator@T_process_to_ptm




def to_Pauli_T_matrix(O):
    '''
    Converts one or two qubit gate matrix to Pauli transfer matrix as ndarray
    
    Args:
        ndarray: gate in ordinary basis (2x2 or 4x4)
        
    Returns:
        ndarray: Pauli transfer matrix (4x4 or 16x16)
    '''
    if (O.shape[0] == 2):
        T = np.identity(4, dtype=complex)
        for i in range(0, 4):
            for j in range(0, 4):
                T[i,j] = 1/2 * np.trace(sigma[i]@O@sigma[j]@np.conj(O.T))
    if (O.shape[0] == 4):
        T = np.identity(16, dtype=complex)
        sigma2d = np.empty((16, 4, 4), np.complex)
        for i in range(0, 4):
            for j in range(0, 4):
                sigma2d[4*i + j,:,:] = np.kron(sigma[i], sigma[j])
        for i in range(0, 16):
            for j in range(0, 16):
                T[i,j] = 1/4 * np.trace(sigma2d[i,:,:]@O@sigma2d[j,:,:]@np.transpose(O.conj()))       
    return np.real(T)




def ptm_fidelity(T1, T2):
    '''
    Calculate fidelity of PTMs.

    Params
    ------
        T1      : PTM
        T2      : PTM

    Returns
    -------
        (np.trace(T1@T2.T) + d)/(d*(d+1))
    '''
    if T1.shape[0] == 4:
        d = 2
    elif T1.shape[0] == 16:
        d = 4
    return (np.trace(T1@T2.T) + d)/(d*(d+1))




def plot_ptm_compare(ptm1, ptm2, dif = False, title1 = '', title2 = '', title3 = ''):
    '''
    Plot PTMs (and compare).

    Params
    ------
        ptm1      : first PTM
        ptm2      : second PTM
        dif       : plot difference? (True/False)
        title1    : title of the first plot
        title2    : title of the second plot
        title3    : title of the third plot
    '''
    matrix1 = np.real_if_close(ptm1)
    matrix2 = np.real_if_close(ptm2)
    figsize_coef = 1
    if dif:
        figsize_coef = 1.5
        if matrix1.shape[0] == 4:
            label = ['I','X','Y','Z']
            fig, (f1, f2, f3) = plt.subplots(
                nrows = 1, ncols = 3,
                figsize=(12*figsize_coef, 4))
        elif matrix1.shape[0] == 16:
            label = ['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']
            fig, (f1, f2, f3) = plt.subplots(
                nrows = 1, ncols = 3,
                figsize=(14*figsize_coef, 5))
        else:
            print('ERROR in size of ptm')
    else:
        if matrix1.shape[0] == 4:
            label = ['I','X','Y','Z']
            fig, (f1, f2) = plt.subplots(
                nrows = 1, ncols = 2,
                figsize=(12*figsize_coef, 4))
        elif matrix1.shape[0] == 16:
            label = ['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']
            fig, (f1, f2) = plt.subplots(
                nrows = 1, ncols = 2,
                figsize=(14*figsize_coef, 5))
        else:
            print('ERROR in size of ptm')

    fontsizes = 12
    cmap_set = 'RdBu'#'bwr'#'RdBu'
    cb = f1.imshow(matrix1, cmap = cmap_set,vmax = 1,vmin = -1)
    fig.colorbar(cb, ax=f1, ticks=[-1, -1/2, 0, 1/2, 1])
    f1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    f1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    f1.set_xticklabels([''] + label, rotation=45, fontsize=fontsizes)
    f1.set_yticklabels([''] + label, fontsize=fontsizes)
    f1.set_title(title1, fontsize=np.around(1.3*fontsizes))

    cb = f2.imshow(matrix2, cmap = cmap_set,vmax = 1,vmin = -1)
    fig.colorbar(cb, ax=f2, ticks=[-1, -1/2, 0, 1/2, 1])
    f2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    f2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    f2.set_xticklabels([''] + label, rotation=45, fontsize=fontsizes)
    f2.set_yticklabels([''] + label, fontsize=fontsizes)
    f2.set_title(title2, fontsize=np.around(1.3*fontsizes))

    if dif:
        m = matrix2-matrix1
        cb = f3.imshow(m, cmap = cmap_set,vmax = np.max(np.abs(m)),vmin = -np.max(np.abs(m)))
        fig.colorbar(cb, ax=f3)
        f3.xaxis.set_major_locator(ticker.MultipleLocator(1))
        f3.yaxis.set_major_locator(ticker.MultipleLocator(1))
        f3.set_xticklabels([''] + label, rotation=45, fontsize=fontsizes)
        f3.set_yticklabels([''] + label, fontsize=fontsizes)
        f3.set_title(title3, fontsize=np.around(1.3*fontsizes))
    plt.show()
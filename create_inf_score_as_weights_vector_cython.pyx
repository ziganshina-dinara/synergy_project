"""
from cython.parallel import prange
import numpy as np
cimport numpy as np
cpdef np.ndarray create_inf_score_as_weights_vector_cython(np.ndarray array_gene_inf_score, np.ndarray space):
    cdef np.ndarray inf_score_as_weights_vector = np.ones(len(space))
    cdef int n = len(space)
    for i in prange(n):
        if space[i] in array_gene_inf_score[:, 0]:
            inf_score_as_weights_vector[i] = array_gene_inf_score[int(np.where(array_gene_inf_score[:, 0] == gene)[0]), 1]
        else:
            inf_score_as_weights_vector[i] = 1
    return inf_score_as_weights_vector
"""
"""
import numpy as np
cpdef np.ndarray create_inf_score_as_weights_vector_cython(np.ndarray array_gene_inf_score, np.ndarray space):

   
    inf_score_as_weights_vector = np.array(list(map(lambda gene: array_gene_inf_score[int(np.where(array_gene_inf_score[:, 0] == gene)[0]), 1] if gene in array_gene_inf_score[:, 0] else 1, space)))
    return inf_score_as_weights_vector
"""
cimport numpy as np
cpdef np.ndarray create_inf_score_as_weights_vector_cython(np.ndarray array_gene_inf_score, np.ndarray space):
    cpdef np.ndarray inf_score_as_weights_vector = np.ones(len(space))
    cpdef int n = len(space)
    for i in range(n):
        if space[i] in array_gene_inf_score[:, 0]:
            inf_score_as_weights_vector[i] = array_gene_inf_score[int(np.where(array_gene_inf_score[:, 0] == space[i])[0]), 1]
        else:
            inf_score_as_weights_vector[i] = 1
    return inf_score_as_weights_vector


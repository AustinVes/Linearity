# Linearity
A Python library that implements mathematics covered in the linear algebra portion of Linearity I from scratch.  
This library provides useful classes and functions for manipulating n-dimensional tensors, matrices, and vectors.  

I designed a novel format for defining tensors, which I refer to as a 'tensor string', because I felt that existing formats like those in MATLAB or Mathematica had design flaws:  
1. Scalars and vectors are conceptually continuous with higher-order tensors but, because lists/arrays (vectors) are commonly given their own specialized syntax in programming languages, existing formats that draw from that syntax make an unnecessary and mathematically inconsistent distinction between specifying scalars and vectors.
2. Existing formats typically force users to define matrices by their row vectors (or whatever the analagous term would be in tensors of rank >2), which is based on how we would organize the data structure behind the scenes so that position (m,n) is at tensor[m][n]. However, we normally talk about matrices in terms of column-vectors, which makes translating between these two formats needlessly confusing. Again, this is unnecessary if we were to just abstract away the presence of nested lists/arrays behind the scenes and simply think in terms of tensor objects.

To address these flaws, I designed tensor strings to allow users to define tensors column-first and with continuous syntax between scalars, vectors, matrices, and higher order tensors.  
The tensor string for the scalar is: `"a"`  
The tensor string for the vector [a,b,c,d] is: `"a,b,c,d"`  
The tensor string for the 2x2 matrix [a&nbsp;&nbsp;c] is: `"a,b,,c,d"`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[b&nbsp;&nbsp;d]  
The tensor string for the 2x2x2 tensor created by "stacking" [e&nbsp;&nbsp;g] on top of that is `"a,b,,c,d,,,e,f,,g,h"`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[f&nbsp;&nbsp;h]  
And so on and so forth...  
This is intended purely as a way for a user to define tensors of any rank and shape.

## What's been implemented so far:
### Tensor class
`Tensor.init(tensor_string)` := initialize a tensor object using a tensor string  
`Tensor.dimensions()` := returns the shape of the tensor  
`Tensor.dimensionality()` := returns the rank of the tensor  
`Tensor.get(x, *y, *z, ...)` := returns the element at position (x,y,z,...)  
`Tensor.set(val, x, *y, *z, ...)` := sets the element at position (x,y,z,...) to val  
### Matrix class  
`Matrix.init(string)` := initialize a matrix object using a tensor string OR a string literal w/ the elements in the shape you want  
`Matrix.view()` := prints the matrix in a readable format
`Matrix.row_vectors()` := returns the row vectors of the matrix as a list of Vector objects  
`Matrix.column_vectors()` := returns the column vectors of the matrix as a list of Vector objects
`Matrix.basis_dimensionality()` := returns the dimensionality of the matrix (i.e. number of columns)
### Augmented_Matrix class  
`Augmented_Matrix.init(A, b)` := initialize an augmented matrix object using a base matrix and an augmenting tensor of rank <3  
`Augmented_Matrix.base_matrix` := returns the base component of the augmented matrix  
`Augmented_Matrix.augment_matrix` := returns the augmented component of the augmented matrix
### Vector class  
`Vector.init(tensor_string)` := initialize a vector object using a tensor string
`Vector.make_column_vector` := returns the vector as a mx1 matrix  
`Vector.make_row_vector` := returns the vector as a 1xn matrix  
`Vector.mag()` := returns the magnitude of the vector  
`Vector.hat()` := returns a unit vector in the direction of the original vector  
### Tensor operations  
`element_wise_operate(A, B, operation)` := perform an element-wise operation of B onto A and return the resulting tensor  
`add(A, B)` := return the element-wise sum of two tensors  
`subtract(A, B)` := return the element-wise difference between two tensors  
`multiply(A, B)` := return the element-wise product of two tensors  
`divide(A, B)` := return the element-wise quotient of two tensors  
`scalar_multiply(A, x)` := return the product of tensor A multiplied by scalar x  
`contract(A, b)` := return the contraction of a tensor by a vector (tensor-by-tensor contraction not yet implemented)  
`swap(A, p1, p2)` := swaps the sub-tensors at p1 and p2 within tensor A and returns the resulting tensor  
### Matrix-specific operations  
`linear_combination(A, b)` := returns the result of linear combination between a matrix and a vector  
`matrix_multiply(A, B)` := returns the product of two matrices  
`transpose(A)` := returns the transpose of matrix A  
### Vector-specific operations
`dot_product(a, b)` := returns a.b
### Template tensor generators  
`make_constant_tensor(val, shape)` := returns a tensor of specified shape where every element is val  
`make_empty_tensor(shape)` := returns a tensor where every element is None  
`make_null_tensor(shape)` := returns a tensor where every element is 0  
### Template matrix generators  
`make_constant_matrix(val, m, n)` := returns an mxn matrix where every element is val  
`make_empty_matrix(m, n)` := returns an mxn matrix where every element is None  
`make_null_matrix(m, n)` := returns an mxn matrix where every element is 0  
`make_identity_matrix(m)` := returns an mxm identity matrix  
### Template vector generators  
`make_constant_vector(val, n)` := returns a vector of length n where every element is val  
`make_null_vector(n)` := returns a vector of length n where every element is 0

### Helper functions that run in the backend
The user should never see these, but they do the heavy lifting to abstract the nested lists into more mathematically correct tensors. Documented here for completion's sake.  
#### Tensor string - matrix string - nested list conversion
`tensor_string_to_nested_list(tensor_string)`  
`nested_list_to_tensor_string(nested_list)`  
`matrix_string_to_tensor_string(matrix_string)`  
#### Tensor/matrix string properties
`matrix_string_or_tensor_string(mystery_string)`  
`get_tensor_string_dimensions(tensor_string)`  
#### Nested list properties
`verify_nested_list(nested_list)`  
`get_nested_list_dimensions(nested_list)`  
#### Nested list manipulation
`multi_index_get(nested_list, indexes, *args)`  
`multi_index_set(val, nested_list, indexes, *args, in_place=True, verify=True)`  
`reverse_nested_list_dimensions(nested_list)`  
`swap_nested_list(nested_list, indexes1, indexes2, in_place=False)`  
#### Template nested list generators
`make_constant_nested_list(val, dimensions, *args, shallowcopy=False)`  
`make_empty_nested_list(dimensions, *args)`  
`make_null_nested_list(dimensions, *args)`  
#### Misc. helper functions
`index_to_pos(indexes, *args)`  
`pos_to_index(positions, *args)`  

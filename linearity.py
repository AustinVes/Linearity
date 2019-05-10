from itertools import groupby
from copy import deepcopy
import operator

# CLASSES
class Tensor:
	def __init__(self, tensor_string):
		
		nested_list = tensor_string_to_nested_list(tensor_string)

		verify_nested_list(nested_list)

		self.vals = nested_list
		return

	def dimensions(self):
		return get_nested_list_dimensions(self.vals)

	def tensor_dimensionality(self):
		return len(self.dimensions())

	def get(self, position, *args):
		if type(position) not in (list, tuple):
			position = [position] + list(args)

		indexes = pos_to_index(position)

		return multi_index_get(self.vals, indexes)

	def set(self, val, position, *args, in_place=True):
		if type(position) not in (list, tuple):
			position = [position] + list(args)

		indexes = pos_to_index(val, self.vals, indexes)

		return multi_index_set(val, self.vals, indexes, in_place=in_place)


class Matrix(Tensor):

	def __init__(self, input_string):

		if matrix_string_or_tensor_string(input_string) is 'matrix_string':
			super().__init__(matrix_string_to_tensor_string(input_string))
		else:
			assert get_tensor_string_dimensions(input_string) <= 2
			super().__init__(input_string)

		return

	def view(self):

		for row in self.vals:
			stringified_row_list = [str(v) for v in row]
			row_string = ' '.join(stringified_row_list)
			print(row_string)

		return

	def row_vectors(self):

		return [Vector(nested_list_to_tensor_string(v)) for v in self.vals]

	def column_vectors(self):

		return [Vector(nested_list_to_tensor_string(v)) for v in transpose(self).vals]

	def basis_dimensionality(self):

		return len(self.row_vector())


class Augmented_Matrix(Matrix):

	def __init__(self, matrix1, matrix2):

		matrix1 = deepcopy(matrix1)

		if type(tensor2) is Matrix:
			matrix2 = deepcopy(tensor2)
		elif type(tensor2) is Vector:
				matrix2 = make_column_vector(tensor2)
		else:
			raise ValueError(f'2nd input must be Matrix or Vector instance, not {type(tensor2)}')

		if matrix1.dimensions()[0] is not matrix2.dimensions()[0]:
			mxn1 = 'x'.join([str(d) for d in matrix1.dimensions()])
			mxn2 = 'x'.join([str(d) for d in matrix2.dimensions()])
			raise ValueError(f'Cannot augment {mxn1} matrix with {mxn2} matrix')

		self.augment_index = matrix1.dimensions()[1]

		matrix1_str = nested_list_to_tensor_string(matrix1.vals)
		matrix2_str = nested_list_to_tensor_string(matrix2.vals)

		augmented_matrix_string = matrix1_str + ',,' + matrix2_str
		super().__init__(augmented_matrix_string)

		return

	def base_matrix(self):

		columns = self.column_vectors()[:self.augment_index]
		transposed_nested_list = [v.vals for v in columns]
		nested_list = transpose(transposed_nested_list)

		return Matrix(nested_list_to_tensor_string(nested_list))

	def augment_matrix(self):

		return Matrix(nested_list_to_tensor_string(self.vals[augment_index:]))


class Vector(Tensor):

	def __init__(self, tensor_string):

		assert get_tensor_string_dimensions(tensor_string) is 1

		super().__init__(tensor_string)

	def make_column_vector(self):

		column_vector = Matrix(nested_list_to_tensor_string(self.vals))
		column_vector.vals = [[val] for val in column_vector.vals]

		return column_vector

	def make_row_vector(self):

		vector_string = nested_list_to_tensor_string(self.vals)
		row_vector_string = ',,'.join(vector_string.split(','))
		row_vector = Matrix(row_vector_string)

		return row_vector()


# TENSOR OPERATIONS
def element_wise_operate(tensor1, tensor2, operation, debug_text=None):

	if not isinstance(tensor1, Tensor) and not isinstance(tensor2, Tensor):
		return operation(tensor1, tensor2)

	if type(tensor1) is not type(tensor2):
		if debug_text:
			raise ValueError(f'Cannot {debug_text[0]} {type(tensor1)} {debug_text[1]} {type(tensor2)}')
		else:
			raise ValueError(f'Cannot compute {operation} on {type(tensor1)} and {type(tensor2)}')

	if tensor1.dimensions() is not tensor2.dimensions():
		mxn1 = 'x'.join([str(d) for d in tensor1.dimensions()])
		mxn2 = 'x'.join([str(d) for d in tensor2.dimensions()])
		raise ValueError(f'Cannot {debug_text[0]} {mxn1} tensor {debug_text[1]} {mxn2} tensor')

	def recursive_operate(tensor1_sublist, tensor2_sublist):

		if type(tensor1_sublist) not in (list, tuple):
			return operation(tensor1_sublist, tensor2_sublist)
		else:
			return [recursive_operate(t1, t2) for t1, t2 in zip(tensor1_sublist, tensor2_sublist)]

	result_nested_list = recursive_operate(tensor1.vals, tensor2.vals)
	result_tensor = type(tensor1)(nested_list_to_tensor_string(result_nested_list))

	return result_tensor

def add(tensor1, tensor2, operation=('add','with')):
	
	return element_wise_operate(tensor1, tensor2, operator.add, debug_text=('add','with'))

def subtract(tensor1, tensor2):

	return element_wise_operate(tensor1, tensor2, operator.sub, debug_text=('subtract','by'))

def multiply(tensor1, tensor2):

	return element_wise_operate(tensor1, tensor2, operator.mul, debug_text=('multiply','by'))

def divide(tensor1, tensor2):

	return element_wise_operate(tensor1, tensor2, operator.div, debug_text=('divide','by'))

def scalar_multiply(tensor, scalar):

	def recursive_multiply(val, nested_sublist):
		if type(nested_sublist) not in (list, tuple):
			return nested_sublist * val
		else:
			return [recursive_multiply(val, s) for s in nested_sublist]

	multiplied_nested_list = recursive_multiply(scalar, tensor.vals)
	multiplied_tensor = type(tensor)(nested_list_to_tensor_string(multiplied_nested_list))

	return multiplied_matrix

def contract(A, b):

	if not isinstance(A, Tensor):
		return A * b

	elif type(A) is Vector:
		if type(b) is Vector:
			return dot_product(A,b)
		elif not isinstance(b, Tensor):
			return scalar_multiply(A,b)
		else:
			raise ValueError(f'Cannot multiply {type(A)} by {type(b)}')

	elif type(A) is Matrix:
		if type(b) is Vector:
			return linear_combination(A, b)
		elif type(b) is Matrix:
			return matrix_multiply(A, b)
		elif not isinstance(b, Tensor):
			return scalar_multiply(A, b)
		else:
			raise ValueError(f'Cannot multiply {type(A)} by {type(b)}')

	else:
		raise ValueError(f'Cannot multiply by n-dimensional tensors, sorry')

def swap(tensor, indexes1, indexes2):

	if type(indexes1) not in (list, tuple):
		indexes1 = [indexes1]
	if type(indexes2) not in (list, tuple):
		indexes2 = [indexes2]

	swapped_nested_list = swap_nested_list(tensor.vals, indexes1, indexes2)

	return type(tensor)(nested_list_to_tensor_string(swapped_nested_list))


# MATRIX-SPECIFIC OPERATIONS
def linear_combination(matrix, vector):

	result_list = [vector.dot(v) for v in matrix.row_vectors()]
	result_vector = Vector(nested_list_to_tensor_string(result_list))

	return result_vector

def matrix_multiply(matrix1, matrix2):

	if matrix1.dimensions()[1] is not matrix2.dimensions()[0]:
		mxn1 = 'x'.join([str(d) for d in matrix1.dimensions()])
		mxn2 = 'x'.join([str(d) for d in matrix2.dimensions()])
		raise ValueError(f'Cannot multiply {mxn1} matrix by {mxn2} matrix')

	result_list = [linear_combination(matrix1, v).vals for v in matrix2.column_vectors()]
	multiplied_matrix = Matrix(nested_list_to_tensor_string(result_list))

	return multiplied_matrix

def transpose(matrix):
	transposed_nested_list = reverse_nested_list_dimensions(matrix.vals)
	transposed_tensor_string = nested_list_to_tensor_string(transposed_nested_list)

	return Matrix(transposed_tensor_string)


# VECTOR-SPECIFIC OPERATIONS
def dot_product(vector1, vector2):
	
	return sum(n1 * n2 for n1, n2 in zip(vector1.vals, vector2.vals))


# TEMPLATE TENSOR GENERATORS
def make_constant_tensor(val, dimensions, *args, shallowcopy=False):
	
	if type(dimensions) not in (list, tuple):
		dimensions = [dimensions] + list(args)

	return Tensor(nested_list_to_tensor_string(make_constant_nested_list(val, dimensions, shallowcopy=shallowcopy)))

def make_empty_tensor(dimensions, *args):
	
	if type(dimensions) not in (list, tuple):
		dimensions = [dimensions] + list(args)

	return Tensor(nested_list_to_tensor_string(make_empty_nested_list(dimensions)))

def make_null_tensor(dimensions, *args):

	if type(dimensions) not in (list, tuple):
		dimensions = [dimensions] + list(args)

	return Tensor(nested_list_to_tensor_string(make_null_nested_list(dimensions)))


# TEMPLATE MATRIX GENERATORS
def make_constant_matrix(val, m, n=None, shallowcopy=False):

	if not n:
		m, n = m

	return Matrix(nested_list_to_tensor_string(make_constant_nested_list(val, m, n, shallowcopy=shallowcopy)))

def make_empty_matrix(m, n=None):

	if not n:
		m, n = m

	return Matrix(nested_list_to_tensor_string(make_empty_nested_list(m, n)))

def make_null_matrix(m, n=None):

	if not n:
		m, n = m

	return Matrix(nested_list_to_tensor_string(make_null_nested_list(m, n)))

def make_identity_matrix(m):
	identity_nested_list = make_null_nested_list(m, m)

	for i in range(m):
		multi_index_set(1, identity_nested_list, m, m)

	return Matrix(nested_list_to_tensor_string(identity_nested_list))


# TEMPLATE VECTOR GENERATORS
def make_constant_vector(val, m, shallowcopy=False):

	return Vector(nested_list_to_tensor_string(make_constant_nested_list(val, m, shallowcopy=shallowcopy)))

def make_null_vector(m):

	return Vector(nested_list_to_tensor_string(make_constant_nested_list(0, m)))


# TENSOR STRING - MATRIX STRING - NESTED LIST CONVERSION
def tensor_string_to_nested_list(tensor_string):
	# assembles a nested list of values from tensor string

	# strip all whitespace
	tensor_string = ''.join(tensor_string.split())

	# get longest occurance of repeated commas
	# represents intended dimensionality of tensor
	nested_list_dimensionality = get_tensor_string_dimensions(tensor_string)

	def recursive_convert(substring, dimension):
		# recursively converts the tensor string into a nested list
		
		if dimension == 0:
			return eval(substring)
		else:
			return [recursive_convert(s, dimension-1) for s in substring.split(',' * dimension)]

	# convert the tensor string into a nested list based on comma sequences
	# not necessarily a valid tensor yet (i.e. lists at the same level may be different lengths)
	# the nested list will be "backwards" because the converter starts from the highest dimension and works down
	reversed_nested_list = recursive_convert(tensor_string, nested_list_dimensionality)
	
	# verify that the nested list is a uniform tensor
	verify_nested_list(reversed_nested_list)

	# correct the dimensional inversion
	nested_list = reverse_nested_list_dimensions(reversed_nested_list)

	return nested_list

def nested_list_to_tensor_string(nested_list):
	reversed_nested_list = reverse_nested_list_dimensions(nested_list)

	def recursive_convert(nested_sublist, dimension):

		if dimension == 0:
			return str(nested_sublist)
		else:
			return (',' * dimension).join([recursive_convert(v, dimension-1) for v in nested_sublist])

	tensor_string = recursive_convert(reversed_nested_list, len(get_nested_list_dimensions(reversed_nested_list)))
	return tensor_string

def matrix_string_to_tensor_string(matrix_string):
	
	row_strings = list(filter(None, matrix_string.split('\n')))
	row_lists = [list(filter(None, row_string.split(' '))) for row_string in row_strings]

	column_lists = reverse_nested_list_dimensions(row_lists)

	column_tensor_strings = [','.join(column_list) for column_list in column_lists]
	if len(column_tensor_strings) > 1:
		tensor_string = ',,'.join(column_tensor_strings)
	else:
		tensor_string = column_tensor_strings[0] + ',,'

	return tensor_string


# TENSOR/MATRIX STRING PROPERTIES
def matrix_string_or_tensor_string(mystery_string):

	if ',' in mystery_string:
		return 'tensor_string'
	else:
		return 'matrix_string'

def get_tensor_string_dimensions(tensor_string):

	return max(len(list(y)) for (c,y) in groupby(tensor_string) if c==',')


# NESTED LIST PROPERTIES
def verify_nested_list(nested_list):
	# recursively verifies that a nested list represents a uniform tensor

	if type(nested_list) not in (list, tuple):
		return 1
	else:
		sublist_sums = [verify_nested_list(n) for n in nested_list]
		if len(set(sublist_sums)) <= 1:
			return sum(sublist_sums)
		else:
			raise ValueError('Malformed tensor: non-uniform nested list tree')

def get_nested_list_dimensions(nested_list):
	# finds the dimensions of a nested list that is a uniform tensor

	verify_nested_list(nested_list)

	def recursive_dimension(nested_sublist):
		# recursively discovers the size of each dimension in nested list

		if type(nested_sublist) not in (list, tuple):
			return []
		else:
			return [len(nested_sublist)] + recursive_dimension(nested_sublist[0])

	return recursive_dimension(nested_list)


# NESTED LIST MANIPULATION
def multi_index_get(nested_list, indexes, *args):
	if type(indexes) not in (list, tuple):
		indexes = [indexes] + list(args)

	def recursive_get(nested_sublist, indexes_sub):
		if len(indexes_sub) is 1:
			return nested_sublist[indexes_sub[0]]
		else:
			return recursive_get(nested_sublist[indexes_sub[0]], indexes_sub[1:])

	return recursive_get(nested_list, indexes)

def multi_index_set(val, nested_list, indexes, *args, in_place=True, verify=True):

	if type(indexes) not in (list, tuple):
		indexes = [indexes] + list(args)

	if not in_place:
		nested_list = deepcopy(nested_list)

	def recursive_set(val, nested_sublist, indexes_sub):
		if len(indexes_sub) is 1:
			nested_sublist[indexes_sub[0]] = val
			return
		else:
			return recursive_set(val, nested_sublist[indexes_sub[0]], indexes_sub[1:])

	recursive_set(val, nested_list, indexes)

	if verify:
		verify_nested_list(nested_list)

	if in_place:
		return
	else:
		return nested_list

def reverse_nested_list_dimensions(nested_list):
	# reverses the order of dimensions of a nested list that is a uniform tensor

	verify_nested_list(nested_list)

	original_dimensions = get_nested_list_dimensions(nested_list)
	reversed_dimensions = list(reversed(original_dimensions))

	reversed_nested_list = make_empty_nested_list(reversed_dimensions)

	def reverse_o_into_r(original_list, reversed_list):
		# populates the second list with the dimensional reverse of the first list
		# double recursive multi-index array manipulation via side effects, holy shit
		# there is absolutely no chance that I will remember how this works in the morning...

		def recursive_original_get(o_sub = original_list, r_i_sub = []):
			if type(o_sub) not in (list, tuple):
				recursive_reversed_set(r_i_sub, o_sub)
			else:
				for i, o in enumerate(o_sub):
					recursive_original_get(o, [i] + r_i_sub)

		def recursive_reversed_set(r_i_sub, o, r_sub = reversed_list):
			if len(r_i_sub) <= 1:
				r_sub[r_i_sub[0]] = o
			else:
				r_i_sub[1:]
				r_i_sub[0]
				recursive_reversed_set(r_i_sub[1:], o, r_sub[r_i_sub[0]])

		recursive_original_get()

		return reversed_list

	reverse_o_into_r(nested_list, reversed_nested_list)

	return reversed_nested_list

def swap_nested_list(nested_list, indexes1, indexes2, in_place=False):
	
	if not in_place:
		nested_list = deepcopy(nested_list)

	a = multi_index_get(nested_list, indexes1)
	b = multi_index_get(nested_list, indexes2)
	multi_index_set(b, nested_list, indexes1)
	multi_index_set(a, nested_list, indexes2)

	if in_place:
		return
	else:
		return nested_list


# TEMPLATE NESTED LIST GENERATORS
def make_constant_nested_list(val, dimensions, *args, shallowcopy=False):
	# creates a nested list where every element is the same

	if type(dimensions) not in (list, tuple):
		dimensions = [dimensions] + list(args)

	def recursive_populate(val, dimensions, shallowcopy):
		# recursively populates the nested list with the value

		if not dimensions:
			if shallowcopy:
				return val
			else:
				return deepcopy(val)
		else:
			return [recursive_populate(val, dimensions[1:], shallowcopy) for _ in range(dimensions[0])]

	constant_nested_list = recursive_populate(val, dimensions, shallowcopy)
	return constant_nested_list

def make_empty_nested_list(dimensions, *args):
	# creates a nested list where every element is None

	return make_constant_nested_list(None, dimensions, *args)

def make_null_nested_list(dimensions, *args):

	return make_constant_nested_list(0, dimensions, *args)	


# MISC. HELPER FUNCTIONS
def index_to_pos(indexes, *args):
	if type(indexes) not in (list, tuple):
		if not args:
			return indexes+1
		else:
			indexes = [indexes] + list(args)
	
	return [n+1 for n in indexes]

def pos_to_index(positions, *args):
	if type(positions) not in (list, tuple):
		if not args:
			return positions-1
		else:
			positions = [positions] + list(args)

	return [n-1 for n in positions



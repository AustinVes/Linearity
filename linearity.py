from copy import deepcopy

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

	def get(position, *args):
		if type(position) not in (list, tuple):
			position = [position] + list(args)

		indexes = pos_to_index(position)

		return multi_index_get(self.vals, indexes)

	def set(val, position, *args, in_place=True):
		if type(position) not in (list, tuple):
			position = [position] + list(args)

		indexes = pos_to_index(val, self.vals, indexes)

		return multi_index_set(val, self.vals, indexes, in_place=in_place)

class Matrix(Tensor):

	def __init__(self, input_string):

		if matrix_string_or_tensor_string(input_string) is 'matrix_string':
			super().__init__(matrix_string_to_tensor_string(input_string))
		else:
			super().__init__(input_string)

		return

	def view(self):

		for row in self.vals:
			stringified_row_list = [str(v) for v in row]
			row_string = ' '.join(stringified_row_list)
			print(row_string)

		return

	def transpose(self):

		transposed_nested_list = reverse_nested_list_dimensions(self.vals)
		transposed_tensor_string = nested_list_to_tensor_string(transposed_nested_list)

		return Matrix(transposed_tensor_string)


def tensor_string_to_nested_list(tensor_string):
	# assembles a nested list of values from tensor string

	from itertools import groupby

	# strip all whitespace
	tensor_string = ''.join(tensor_string.split())

	# get longest occurance of repeated commas
	# represents intended dimensionality of tensor
	nested_list_dimensionality = max(len(list(y)) for (c,y) in groupby(tensor_string) if c==',')

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

def matrix_string_or_tensor_string(mystery_string):

	if ',,' in mystery_string:
		return 'tensor_string'
	else:
		return 'matrix_string'

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

	return [n-1 for n in positions]

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

def swap(nested_list, indexes1, indexes2, in_place=False):
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
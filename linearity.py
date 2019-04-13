class Tensor:
	def __init__(self, tensor_string):
		
		nested_list = tensor_string_to_nested_list(tensor_string)

		verify_nested_list(nested_list)

		self.vals = nested_list
		self.dimensions = get_nested_list_dimensions(self.vals)
		self.tensor_dimensionality = len(self.dimensions)


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

def verify_nested_list(nested_list):
	# recursively verifies that a nested list represents a uniform tensor

	if type(nested_list) is not list:
		return 1
	else:
		sublist_sums = [verify_nested_list(n) for n in nested_list]
		if len(set(sublist_sums)) <= 1:
			return sum(sublist_sums)
		else:
			raise ValueError('Malformed tensor: unbalanced nested list tree')

def get_nested_list_dimensions(nested_list):
	# finds the dimensions of a nested list that is a uniform tensor

	verify_nested_list(nested_list)

	def recursive_dimension(nested_sublist):
		# recursively discovers the size of each dimension in nested list

		if type(nested_sublist) is not list:
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
			if type(o_sub) is not list:
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

def make_constant_nested_list(val, dimensions, *args, shallowcopy=False):
	# creates a nested list where every element is the same

	from copy import deepcopy

	if type(dimensions) is not list:
		dimensions = [dimensions] + args

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
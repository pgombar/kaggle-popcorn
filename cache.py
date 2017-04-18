import cPickle, gzip, sys

# serializes data and saves in a (somewhat) compressed format
def save_zipped_pickle(obj, filename, protocol=-1):
	with gzip.open(filename, 'wb') as f:
		cPickle.dump(obj, f, protocol)
		f.close()

# loads compressed data and restores original state
def load_zipped_pickle(filename):	# loads and unpacks
	with gzip.open(filename, 'rb') as f:
		loaded_object = cPickle.load(f)
		f.close()
		return loaded_object
	
# Automaticly caches a function's value with given arguments
def auto_cache(function, *args):
	
	cs = "\\/?\":<>*|"
	
	def _clear_str(str):
		for c in cs:
			str = str.replace(c, "")
		return str
		
	s = 'cache/' + _clear_str(str(function) + str(args)) + '.bin';
	if os.path.isfile(s):
		ret_val = load_zipped_pickle(s);
	else:
		ret_val = function(args);
		save_zipped_pickle(ret_val, s);
	return ret_val

import functools

def lazy_property(function):
	attribute= '_cache_' + function.__name__

	@property
	@function.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator	
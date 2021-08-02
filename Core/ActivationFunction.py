from Core.Defs import *

class ActivationFunction():
	
	def __init__(self,func,dfunc,name):
		self.func  = func
		self.dfunc = dfunc
		self.name  = name
	
	def get(activation_function = "sigmoid"):
		table = {
			"sigmoid" : ActivationFunction.sigmoid(),
			"tanh"    : ActivationFunction.tanh(),
			"relu"    : ActivationFunction.relu()
		}
		return table[activation_function]
	
	@staticmethod	
	def sigmoid():
		func  = lambda x : (1 / (1 + math.exp(-x) ))  
		dfunc = lambda y : (y * (1 - y))
		return ActivationFunction(func,dfunc,'sigmoid')
	
	@staticmethod	
	def tanh():
		func  = lambda x : (math.tanh(x))
		dfunc = lambda y : (1 - (y * y))
		return ActivationFunction(func,dfunc,'tanh')

	@staticmethod	
	def relu():
		func  = lambda x : ( x if x >= 0 else 0)
		dfunc = lambda y : ( 1 if y >= 0 else 0)
		return ActivationFunction(func,dfunc,'relu')




from Core.ActivationFunction import *


class Perceptron:
	
	def __init__(self,nodes = 2):
		self.nodes = nodes
		self.weights = [random.random() - 0.5 for node in range(self.nodes)]
		self.bias = random.random() - 0.5
		self.set_activation_function()
		self.set_learning_rate()
	
	def set_activation_function(self,activation_function = ActivationFunction.sigmoid()):
		self.activation_function = activation_function
	
	def set_learning_rate(self,learning_rate = 0.1):
		self.learning_rate = learning_rate
		
	def predict(self,layers_i = []):
		output = self.bias
		output = output + sum(list(map(lambda x : x[0] * x[1],list(zip(layers_i,self.weights)))))
		output = self.activation_function.func(output)
		return output
	
	def learn(self,layers_i = [],target = 0):
		output = self.predict(layers_i)
		error = (target - output)
		gradient = error * self.activation_function.dfunc(output) * self.learning_rate
		self.bias = self.bias + gradient
		self.weights = list(map(lambda x : x[1] + x[0] * gradient,list(zip(layers_i,self.weights))))
		return error
	
	def train(self,datasets = [],epochs = 10000,verbose = False):
		delta = int(epochs * 0.05)
		epoch = 1
		if verbose:
			time_start = datetime.now()
		while epoch <= epochs:
			dataset = random.choice(datasets)
			layers_i = dataset[:-1]
			if len(layers_i) != self.nodes:
				raise Exception("Input must be of length {}.".format(self.nodes))
			target = dataset[-1]
			error = self.learn(layers_i,target)			
			if verbose:
				if epoch % delta == 0:
					print("{:3}% Completed Error: {}".format(int(epoch * 100 / epochs),error))
			epoch = epoch + 1
		if verbose:
			time_end = datetime.now()
			elapsed  = time_end - time_start
			print("\nEpoch: {} Elapsed : {} seconds.\n".format(epoch - 1,elapsed))
		return self
	
	def to_json(self):
		obj = '{\n'
		obj = obj + '\t"nodes" : ' + str(self.nodes) + ',\n' 
		obj = obj + '\t"weights" : ' + str(self.weights) + ',\n' 
		obj = obj + '\t"bias" : ' + str(self.bias) + ',\n' 
		obj = obj + '\t"activation_function" : '+ "'{}'".format(self.activation_function.name) + ',\n' 
		obj = obj + '\t"learning_rate" : '+ str(self.learning_rate) + '\n' 
		obj = obj + '}\n' 
		return obj
	
	@staticmethod
	def to_file(path,p):
		path = "{}\\{}".format(os.getcwd(),path)
		f = open(path,"w")
		f.write(p.to_json())
		f.close()
		return True
	
	@staticmethod
	def from_file(path):
		path = "{}\\{}".format(os.getcwd(),path)
		f = open(path,"r")
		contents = f.read()
		f.close()
		contents = contents.replace('\n',' ') 
		obj = ast.literal_eval(contents)
		p = Perceptron(obj["nodes"])
		p.weights = obj["weights"]
		p.bias = obj["bias"]
		p.set_activation_function(ActivationFunction.get(obj["activation_function"]))
		p.set_learning_rate(obj["learning_rate"])
		return p
	

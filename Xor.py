from Core.Perceptron import *

models = []

def train():
	datasets = [
		[0,0,0],
		[0,1,1],
		[1,0,0],
		[1,1,0]
	]
	p = Perceptron(2)
	p.train(datasets)
	models.append(p)
	datasets = [
		[0,0,0],
		[0,1,0],
		[1,0,1],
		[1,1,0]
	]
	p = Perceptron(2)
	p.train(datasets)
	models.append(p)

def tests():
	datasets = [
		[0,0,0],
		[0,1,1],
		[1,0,1],
		[1,1,0],
	]
	for dataset in datasets:
		layers_i = dataset[:-1]
		target = dataset[-1]
		output = float("-inf")
		for model in models:
			value = round(model.predict(layers_i))
			if value > output:
				output = value
		print("layers_i: {} target: {} output: {}".format(layers_i,target,output))

def main():
	train()
	tests()
	
main()
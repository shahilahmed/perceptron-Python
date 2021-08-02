from Core.Perceptron import *

datasets = [
	[0,0,0,1],
	[0,0,1,0],
	[0,1,0,0],
	[0,1,1,0],
	[1,0,0,0],
	[1,0,1,0],
	[1,1,0,0],
	[1,1,1,0]
]

is_train = True
is_train = False
if is_train:
	p = Perceptron(3)
	p.train(datasets,10000)
	Perceptron.to_file("Models\\Nand.json",p)
else:
	p = Perceptron.from_file("Models\\Nand.json")
	for dataset in datasets:
		layers_i = dataset[:-1]
		target = dataset[-1]
		output = p.predict(layers_i)
		print("layers_i: {} target: {} output: {}".format(layers_i,target,round(output)))
	print()

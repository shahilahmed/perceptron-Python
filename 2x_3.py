from Core.Perceptron import *

def f(x):
	return 2 * x + 3

datasets = [
	[1,5],
	[2,7],
	[3,9],
	[4,11],
	[5,13]
]
name = "2x_3"
nodes = (len(datasets[0]) - 1)

def train():
	p = Perceptron(nodes)
	p.set_activation_function(ActivationFunction.relu())
	p.train(datasets,10000,True)
	Perceptron.to_file("Models\\{}.json".format(name),p)

def test():
	p = Perceptron.from_file("Models\\{}.json".format(name))
	datasets = list(range(50000))
	for count in range(10):
		layers_i = [random.choice(datasets)]
		target = f(layers_i[0])
		output = round(p.predict(layers_i))
		print("layers_i: {} target: {} output: {}".format(layers_i,target,output))

def main():
	argv = copy.deepcopy(sys.argv)
	argv.pop(0)
	if argv:
		command = argv[0]
		if command == "-train":
			train()
		elif command == "-test":
			test()
		else:
			train()
			test()
	else:
		train()
		test()

main()
import { range, bipolarStep, random, copy } from './utils.js'

const Perceptron = (nInputs, learningRate, maxEpochs) => ({
	weights: range(nInputs + 1).map(() => random(-1, 1)),
	activation: 'bipolarStep'
})

const activate = (weights, input) => {
	if (input.length <= 0) {
		return 0
	}

	return input[0] * weights[0] + activate(weights.slice(1), input.slice(1))
}

const train = ({ perceptron, inputs, targets, learningRate, maxEpochs }) => {
	let existsError = undefined
	let weights = copy(perceptron.weights)
	let epochs = 0

	do {
		epochs += 1
		existsError = false

		inputs.forEach((input, i) => {
			const [ x1, x2 ] = input
			const [ bias, w1, w2 ] = weights
			const inputWithBias = [-1].concat(input)
			const output = bipolarStep(activate(weights, inputWithBias))
			const error = targets[i] - output
			const weightedError = learningRate * error

			if (error) {
				existsError = true
				weights = [
					bias + weightedError * -1,
					w1 + weightedError * x1,
					w2 + weightedError * x2
				]
			}
		})
	} while (existsError && epochs < maxEpochs)

	return {
		perceptron: {
			...perceptron,
			weights
		},
		epochs
	}
}

export default Perceptron

export {
	train
}
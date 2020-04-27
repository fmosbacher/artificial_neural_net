const range = (min = 0, max) => {
	if (!max) {
		max = min
		min = 0
	}

	return min < max ? [min].concat(range(min + 1, max)) : []
}

const bipolarStep = value => value > 0 ? 1 : -1

const random = (min, max) => {
	if (!max) {
		max = min
		min = 0
	}

	return Math.random() * (max - min) + min
}

const copy = data => JSON.parse(JSON.stringify(data))

const dot = (m1, m2) => {
	
}

export {
	range,
	bipolarStep,
	random,
	copy
}
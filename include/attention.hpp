#pragma once

#include <stdlib.h>

class Tensor {
	private:
		size_t shape;
	public:
		size_t get_shape() {return shape;}
};

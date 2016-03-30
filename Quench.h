#ifndef QUENCH_H
#define QUENCH_H

enum QUENCHTYPE {NONE, FREEZE, LINEAR, EXPONENTIAL, STEP};

struct QUENCH {
	QUENCHTYPE type;
	double init_beta;
	double final_beta;

	double lin_gradient;
	double exp_exponent;

	double step_delta_beta;
	std::uint32_t step_period;

	QUENCH() {
		type = QUENCHTYPE::NONE;
		init_beta = 0;
		final_beta = 10;
		lin_gradient = 0;
		exp_exponent = 0;
		step_delta_beta = 0;
		step_period = 0;
	}

	double operator()(const double t);
	size_t time_taken();
};

#endif
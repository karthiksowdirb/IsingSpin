#include <cstdint>
#include <cmath>

#include "Quench.h"

double freeze(const double final_beta) {
	return final_beta;
}

std::uint32_t freeze_time() {
	return 1;
}

double linear(const double init_beta, const double gradient, const double time_step) {
	return gradient * time_step + init_beta;
}

std::uint32_t linear_time(const double init_beta, const double final_beta, const double gradient) {
	return (final_beta - init_beta) / gradient;
}

double exponential(const double init_beta, const double exponent, const double time_step) {
	return init_beta * std::exp(exponent * time_step);
}

std::uint32_t exponential_time(const double init_beta, const double final_beta, const double exponent) {
	return log(final_beta / init_beta) / exponent;
}

double step(const double init_beta, const double delta_beta, const std::uint32_t period, const double time_step) {
	return init_beta + ((int)time_step / period) * delta_beta;
}

std::uint32_t step_time(const double init_beta, const double final_beta, const double delta_beta, const uint32_t period) {
	return ((final_beta - init_beta) / delta_beta) * period;
}

double QUENCH::operator()(const double t) {
	switch (type) {
	case QUENCHTYPE::LINEAR:
		return linear(init_beta, final_beta, t);
		break;
	case QUENCHTYPE::EXPONENTIAL:
		return exponential(init_beta, exp_exponent, t);
		break;
	case QUENCHTYPE::STEP:
		return step(init_beta, step_delta_beta, step_period, t);
		break;
	case QUENCHTYPE::FREEZE:
	default:
		return freeze(final_beta);
		break;
	}
}

size_t QUENCH::time_taken() {
	switch (type) {
	case QUENCHTYPE::LINEAR:
		return linear_time(init_beta, final_beta, lin_gradient);
		break;
	case QUENCHTYPE::EXPONENTIAL:
		return exponential_time(init_beta, final_beta, exp_exponent);
		break;
	case QUENCHTYPE::STEP:
		return step_time(init_beta, final_beta, step_delta_beta, step_period);
		break;
	case QUENCHTYPE::FREEZE:
	default:
		return freeze_time();
		break;
	}
}
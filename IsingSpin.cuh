#ifndef ISING_SPIN_CUH
#define ISING_SPIN_CUH

#include <cstdint>
#include <functional>
#include "../../../FileIO.h"
#include "Quench.h"
#include <cuda.h>
#include <curand_kernel.h>

class IsingSpin {
private:
	std::uint32_t N;
	std::uint32_t maxdegree;

	std::uint16_t *d_adjlist;
	std::uint16_t *d_deglist;

	curandState *randStates;

public:
	IsingSpin() {
		d_adjlist = nullptr;
		d_deglist = nullptr;
		randStates = nullptr;
	}

	~IsingSpin() {
		cudaFree(d_deglist);
		cudaFree(d_adjlist);
		cudaFree(randStates);
	}

	void set_graph(GRAPH &graph);
	
	void run(std::int8_t *d_spins, const std::uint32_t mix_time, const std::uint32_t quenching_time, const std::uint32_t relaxation_time, const QUENCH quench);
};

#endif
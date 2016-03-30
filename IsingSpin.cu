#include <cstdint>
#include <cuda.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>

#include "IsingSpin.cuh"
#include "Quench.h"

#define BOLTZMANN 1.3806488e-23

__global__ void rand_setup_kernel(curandState *randStates, const int N, unsigned long long int seed);
__global__ void spin_setup_kernel(std::uint8_t *d_spins, curandState *randStates, const int N);
__global__ void Ising_Spin_kernel(std::uint8_t *d_spins, curandState *randStates, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, const int N, const int maxdegree, const double beta, const int time_step);

void IsingSpin::set_graph(GRAPH &graph) {
	N = graph.NoV;
	maxdegree = graph.max_deg;

	//Create arrays to store the graph in the device memory.
	cudaFree(d_deglist);
	cudaFree(d_adjlist);

	cudaMalloc((void**)&d_deglist, sizeof(std::uint16_t)*N);
	cudaMalloc((void**)&d_adjlist, sizeof(std::uint16_t)*N*maxdegree);

	//Copy graphs to device memory.
	cudaMemcpy(d_deglist, graph.deglist, sizeof(std::uint16_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjlist, graph.adjlist, sizeof(std::uint16_t)*N*maxdegree, cudaMemcpyHostToDevice);

	//Crate an array of random number generators to be used in the ising spin kernel.
	cudaMalloc((void**)&randStates, sizeof(curandState)*N);
}

__global__ void rand_setup_kernel(curandState *randStates, const int N, unsigned long long int seed) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		curand_init(seed, n, 0, &randStates[n]);
	}
}

__global__ void spin_setup_kernel(std::int8_t *d_spins, curandState *randStates, const int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		double x = curand_uniform_double(&randStates[n]);
		if (x < 0.5) {
			d_spins[N+n] = -1;
		}
		else {
			d_spins[N+n] = 1;
		}
	}
}

__global__ void Ising_Spin_kernel(std::int8_t *d_spins, curandState *randStates, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, const int N, int maxdegree, const double beta, const int time_step) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		int n_spin = (int)d_spins[N*time_step + n];
		int k = (int)d_deglist[n];

		int same_spin_count = 0;
		
		int j = 0;
		int s = 0;
		for (int i = 0; i < k; i++) {
			j = (int)d_adjlist[n*maxdegree + i];
			s = (int)d_spins[N*time_step + j];
			if (s == n_spin) {
				same_spin_count++;
			}
		}

		const double x = -2. * beta * (2. * (double)same_spin_count - k);
		const double y = exp((double)x);
		const double p = y / (1. + y);
		double r = curand_uniform_double(&randStates[n]);

		if (r < p) {
			d_spins[N*time_step + N + n] = -n_spin;
		}
		else {
			d_spins[N*time_step + N + n] = n_spin;
		}
	}
}

void IsingSpin::run(std::int8_t *d_spins, const std::uint32_t mix_time, const std::uint32_t quenching_time, const std::uint32_t relaxation_time, QUENCH quench) {
	const double mix_beta = quench.init_beta;
	const double relax_beta = quench.final_beta;

	std::uint32_t time_step = 0;

	size_t blocksize = 256;
	size_t blocknum = (N / blocksize) + 1;
	
	//Initialise the random number generators.
	std::random_device rd;
	long long int seed = rd();
	rand_setup_kernel<<<blocknum, blocksize>>>(randStates, N, seed);

	//Set each vertex to a random spin.
	spin_setup_kernel<<<blocknum, blocksize>>>(d_spins, randStates, N);

	time_step++;

	//Mixing of the opinions. Opinions can change freely for a long time.
	for (size_t t = 0; t < mix_time; t++) {
			Ising_Spin_kernel<<<blocknum, blocksize>>>(d_spins, randStates, d_deglist, d_adjlist, N, maxdegree, mix_beta, time_step);
		time_step++;
	}

	//Quenching of the system. Beta is changed (increased) until reaching a final beta value.
	double beta;
	for (size_t t = 0; t < quenching_time; t++) {
		beta = quench(t);
			Ising_Spin_kernel<<<blocknum, blocksize>>>(d_spins, randStates, d_deglist, d_adjlist, N, maxdegree, beta, time_step);
		time_step++;
	}

	//Relaxation of the system. Vertices can change opinion for a time so that an equilibrium or steady state can be reached.
	for (size_t t = 0; t < relaxation_time-1; t++) {
			Ising_Spin_kernel<<<blocknum, blocksize>>>(d_spins, randStates, d_deglist, d_adjlist, N, maxdegree, relax_beta, time_step);
		time_step++;
	}
}
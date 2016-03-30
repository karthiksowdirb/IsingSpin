#include "Analysers.cuh"
#include "../../../FileIO.h"

#include <cstdint>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <queue>
#include <thread>
#include <fstream>
#include <cmath>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void apsp_thread(std::int32_t *g_apsp, GRAPH &h_graph, const size_t N, const size_t n) {
	const size_t Nn = N*n;
	std::queue<std::uint16_t> q;
	q.push(n);
	g_apsp[Nn + n] = 0;
	std::uint16_t j;
	std::uint16_t m;
	while (q.size()) {
		m = q.front();
		q.pop();

		for (size_t k = 0; k < h_graph.deglist[m]; k++) {
			j = h_graph.adjlist[m*h_graph.max_deg + k];

			if (g_apsp[Nn + j] < 0) {
				q.push(j);
				g_apsp[Nn + j] = g_apsp[Nn + m] + 1;
			}
		}
	}
}

void graph_apsp(std::int32_t *g_apsp, GRAPH &h_graph) {
	const size_t N = h_graph.NoV;
	std::fill(g_apsp, g_apsp + N*N, -1);

	std::queue<std::uint16_t> q;
	std::uint16_t j;
	std::uint16_t m;

	for (size_t n = 0; n < N; n++) {
		size_t Nn = N*n;
		q.push(n);

		while (q.size()) {
			m = q.front();
			q.pop();

			for (size_t k = 0; k < h_graph.deglist[m]; k++) {
				j = h_graph.adjlist[m*h_graph.max_deg + k];

				if (g_apsp[Nn + j] < 0) {
					q.push(j);
					g_apsp[Nn + j] = g_apsp[Nn + m] + 1;
				}
			}
		}

	}
}

__global__ void running_average_kernel(std::int8_t *d_spin, const size_t N, const size_t T, const size_t window) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		/*
			- If the spin of a vertex flips a lot then it is unstable.
			- If it does not flip above a threshold of 0.4 then it is a stable vertex.
				- We find the average magnetisation of the vertex as a sum of its spins
				  over a period of time.
				- If the magnitude of the magnetisation over a period of time is above 1/3 then
				  we set the value of the spin to the average.
				- If the mangitude of the magnetisation over a period of time is less than 1/3 then
				  we do not change the value of the spin.
		*/

		const double thresh = 0.5;
		const double magThresh = 1. / 3.;
		double sum = 0;
		double flips = 0;
		for (int t = window; t < T-window-1; t++) {
			sum = 0;
			for (int w = t - window; w <= t + window; w++) {
				sum += d_spin[N*w + n];
				if (d_spin[N*w + n] != d_spin[N*w + N + n]) {
					flips++;
				}
			}

			sum /= 2. * (double) window + 1.;
			flips /= 2. * (double)window;

			if (flips > thresh) {
				d_spin[N*t + n] = 0;
			}
			else if (sum > magThresh) {
				d_spin[N*t + n] = 1;
			}
			else if (sum < -magThresh) {
				d_spin[N*t + n] = -1;
			}
		}
	}
}

void noise_remover(std::int8_t *d_spin, const size_t N, const size_t T, const size_t window) {
	size_t blocksize = 256;
	size_t blocknum = (N / blocksize) + 1;

	running_average_kernel<<<blocknum,blocksize>>>(d_spin, N, T, window);
}

__global__ void magnetisation_graph_kernel(std::int16_t *d_mag, std::int8_t *d_spin, const size_t N, const size_t T) {
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t < T) {
		int s = 0;
		for (int n = 0; n < N; n++) {
			s += d_spin[N*t + n];
		}
		d_mag[t] = s;
	}
}

void magnetisation(MAGNETISATION &mag, std::int8_t *d_spin) {
	const size_t N = mag.N;
	const size_t T = mag.T;

	size_t blocksize = 256;
	size_t blocknum = (T / blocksize) + 1;

	std::int16_t *d_mag;
	cudaMalloc((void**)&d_mag, sizeof(std::int16_t)*T);

	cudaMemcpy(d_mag, mag.h_gmags, sizeof(std::int16_t)*T, cudaMemcpyHostToDevice);

		magnetisation_graph_kernel<<<blocknum, blocksize>>>(d_mag, d_spin, N, T);

	cudaMemcpy(mag.h_gmags, d_mag, sizeof(std::int16_t)*T, cudaMemcpyDeviceToHost);

	cudaFree(d_mag);
}



void componentBFS(std::vector<std::uint16_t> &sizes, std::int16_t *memberships, std::uint32_t N, std::int8_t *h_spins, GRAPH &h_graph) {
	std::fill(memberships, memberships + N, -1);

	std::uint16_t size = 0;
	std::uint16_t comp = 0;

	std::queue<std::uint16_t> q;

	std::uint16_t m, n;
	for (size_t i = 0; i < N; i++) {
		if (memberships[i] < 0) {
			//printf("\t\t\t i = %d\n", i);
			size = 0;
			q.push(i);
			memberships[i] = comp;
			while (q.size()) {
				m = q.front();
				q.pop();
				size++;

				//printf("\t\t\t\t m = %d | K = %d\n", m, h_graph.deglist[m]);

				for (size_t k = 0; k < h_graph.deglist[m]; k++) {
					n = h_graph.adjlist[m*h_graph.max_deg + k];
					if (memberships[n] < 0 && h_spins[m] == h_spins[n]) {
						memberships[n] = comp;
						q.push(n);
					}
				}
			}

			sizes.push_back(size);
		}
		comp++;		
	}
}

void crowdSizeDists(jaggedlist<std::uint16_t> &h_crowdsizes, std::int8_t *h_spins, GRAPH &h_graph, const size_t N, const size_t T, const size_t interval) {
	std::int16_t *memberships = new std::int16_t[N]();


	size_t step = 0;
	for (size_t t = 0; t < T; t += interval) {
		h_crowdsizes.set_size(step+1);
		componentBFS(h_crowdsizes[step], memberships, N, h_spins+t*N, h_graph);
		//printf("\t\tt=%d | s = %d\n", t, h_crowdsizes[step].size());
		step++;
	}

	delete[] memberships;
}

std::int32_t identify_features(std::int32_t *h_features, std::int8_t *h_spins, GRAPH &h_graph, const size_t N, const size_t T) {
	struct dim2 {
		std::uint16_t v;
		std::uint16_t t;

		void operator()(const std::uint16_t v, const std::uint16_t t) {
			this->v = v;
			this->t = t;
		}
	};
	
	std::fill(h_features, h_features + N*T, -1);
	std::int32_t f = -1;
	dim2 x, y;

	std::int8_t s;

	std::queue<dim2> q;

	std::uint16_t m, K;
	
	size_t vcount;

	std::uint16_t t1, t2;

	for (size_t t = 0; t < T; t++) {
		for (size_t i = 0; i < N; i++) {
			if (h_features[N*t + i] < 0) {
				f++;
				h_features[N*t + i] = f;
				x(i, t);
				q.push(x);
				s = h_spins[N*t + i];
				vcount = 1;
				t1 = t2 = (std::uint16_t) t;
				while (q.size()) {
					x = q.front();
					q.pop();
					vcount++;

					if (x.t < t1) { t1 = x.t; }
					if (x.t > t2) { t2 = x.t; }

					K = h_graph.deglist[x.v];
					for (std::uint16_t k = 0; k < K; k++) {
						m = h_graph.adjlist[h_graph.max_deg*x.v + k];
						y(m, x.t);
						if (h_features[N*y.t + y.v] < 0 && h_spins[N*y.t + y.v] == s) {
							h_features[N*y.t + y.v] = f;
							q.push(y);
						}
					}

					if (x.t > 0) {
						if (h_features[N*(x.t - 1) + x.v] < 0 && h_spins[N*(x.t - 1) + x.v] == s) {
							h_features[N*(x.t - 1) + x.v] = f;
							y(x.v, (x.t - 1));
							q.push(y);
						}
					}

					if (x.t < T-1) {
						if (h_features[N*(x.t + 1) + x.v] < 0 && h_spins[N*(x.t + 1) + x.v] == s) {
							h_features[N*(x.t + 1) + x.v] = f;
							y(x.v, (x.t + 1));
							q.push(y);
						}
					}
				}
				//h_lifetimes.push_back(1 + t2 - t1);
			}
		}
	}

	return f;
}

__global__ void membership_pathlength_kernel(float *d_memberships_apl, float *d_memberships_mpl, std::uint16_t *d_apsp, const std::uint32_t N, const std::uint32_t t) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		int iN = i*N;
		float plcount = 0;
		float plave = 0;
		float plmax = 0;
		for (int j = 0; j < N; j++) {
			if (d_apsp[iN + j] < N) {
				plcount++;
				plave += (float)d_apsp[iN + j];
				if (plmax < d_apsp[iN + j]) {
					plmax = d_apsp[iN + j];
				}
			}
		}

		plave /= plcount;

		d_memberships_apl[N*t + i] = plave;
		d_memberships_mpl[N*t + i] = plmax;
	}
}

//oid membershipBFS_thread(std::int16_t *h_memberships, std::uint16_t *h_apsp, std::int8_t *h_spins, GRAPH &h_graph, const size_t N, const size_t t)
void membershipBFS_thread(std::int16_t *h_memberships, std::uint16_t *h_membership_count, std::uint16_t *h_apsp, std::int8_t *h_spins, GRAPH *h_graph_ptr, const size_t N, const size_t t) {
	GRAPH &h_graph = *h_graph_ptr;
	const std::uint16_t inf = (std::uint16_t) -1;

	//Variable required to reduce calculation operations.
	size_t NT = N*t;
	size_t iN;

	std::int8_t s;
	std::uint16_t m, n;
	std::int16_t mbrid = -1;
	
	std::queue<std::uint16_t> q;
	h_membership_count[t] = 0;
	for (size_t i = 0; i < N; i++) {
		
		//If a vertex dow not have a membership id then it is a new crowd.
		if (h_memberships[NT + i] < 0) {
			h_membership_count[t]++;
			s = h_spins[NT + i];
			mbrid++;
			//printf("\t\t\t i = %d\n", i);
			q.push(i);
			h_memberships[NT + i] = mbrid;

			iN = i*N;
			while (q.size()) {
				m = q.front();
				q.pop();

				//printf("\t\t\t\t m = %d | K = %d\n", m, h_graph.deglist[m]);

				for (size_t k = 0; k < h_graph.deglist[m]; k++) {
					n = h_graph.adjlist[m*h_graph.max_deg + k];
					if (h_memberships[NT + n] < 0 && h_spins[NT + n] == s) {
						h_memberships[NT + n] = mbrid;
						q.push(n);
					}
				}
			}
		}
	}

	//Set all pathlengths to infinity.
	std::fill(h_apsp, h_apsp + N*N, inf);

	//Use BFS to calculate the APSP of every vertex.
	for (size_t i = 0; i < N; i++) {
		iN = i*N;

		h_apsp[iN + i] = 0;
		mbrid = h_memberships[NT + i];
		q.push(i);

		while (q.size()) {
			m = q.front();
			q.pop();

			for (size_t k = 0; k < h_graph.deglist[m]; k++) {
				n = h_graph.adjlist[m*h_graph.max_deg + k];
				if (h_memberships[NT + n] == mbrid) {
					if (h_apsp[iN + n] == inf) {
						h_apsp[iN + n] = h_apsp[iN + m] + 1;
						q.push(n);
					}
				}
			}
		}
	}
}

void membershipBFS(std::int16_t *h_memberships, std::uint16_t *h_membership_count, float *h_memberships_apl, float *h_memberships_mpl, std::int8_t *h_spins, GRAPH &h_graph, const std::uint32_t N, const std::uint32_t T, const std::uint32_t mixing_time) {
	//cuda kernel block variables.
	size_t blocksize = 256;
	size_t blocknum = (N / blocksize) + 1;

	const size_t numOfThr = 7;
	std::thread bfs_threads[numOfThr];
	//Data arrays required for the APSP.
	std::uint16_t **h_apsps = new std::uint16_t*[numOfThr];
	for (size_t thrid = 0; thrid < numOfThr; thrid++) {
		h_apsps[thrid] = new std::uint16_t[N*N]();
	}

	std::uint16_t *d_apsp;
	cudaMalloc((void**)&d_apsp, sizeof(std::uint16_t)*N*N);

	//Data arrays required for computing the average pathlength and max path length of every vertex.
	std::int16_t *d_memberships;
	float *d_memberships_apl, *d_memberships_mpl;
	cudaMalloc((void**)&d_memberships, sizeof(std::int16_t)*N);
	cudaMalloc((void**)&d_memberships_apl, sizeof(float)*N*T);
	cudaMalloc((void**)&d_memberships_mpl, sizeof(float)*N*T);

	for (size_t t = mixing_time; t < T; t += numOfThr) {
		if (t % (numOfThr*10) == 0) {
			printf("t: %d / %d\n", t, T - 1);
		}

		//Launch threads threads to do both BFS'.
		for (size_t thrid = 0; thrid < numOfThr; thrid++) {
			if (t + thrid < T) {
				bfs_threads[thrid] = std::thread(membershipBFS_thread, h_memberships, h_membership_count, h_apsps[thrid], h_spins, &h_graph, N, t + thrid);
			}
		}


		//Join the the threads and comput the apl's and mpl's for each vertex.
		for (size_t thrid = 0; thrid < numOfThr; thrid++) {
			if (t + thrid < T) {
				bfs_threads[thrid].join();
				cudaMemcpy(d_apsp, h_apsps[thrid], sizeof(std::uint16_t)*N*N, cudaMemcpyHostToDevice);
				membership_pathlength_kernel << <blocknum, blocksize >> >(d_memberships_apl, d_memberships_mpl, d_apsp, N, t + thrid);
			}
		}
	}

	cudaMemcpy(h_memberships_apl, d_memberships_apl, sizeof(float)*N*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_memberships_mpl, d_memberships_mpl, sizeof(float)*N*T, cudaMemcpyDeviceToHost);

	for (size_t thrid = 0; thrid < numOfThr; thrid++) {
		delete[] h_apsps[thrid];
	}
	delete[] h_apsps;
	cudaFree(d_apsp);
	cudaFree(d_memberships);
	cudaFree(d_memberships_apl);
	cudaFree(d_memberships_mpl);
}

__global__ void get_crowd_kernel(float *d_crowd, std::int16_t *d_memberships, std::uint32_t N, std::uint32_t t, std::int16_t membership) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		d_crowd[n] = (float) 0.;

		if (d_memberships[N*t + n] == membership) {
			//printf("\n[%d]: %d | %d: %d", n, d_memberships[N*t + n], membership, (int)(d_memberships[N*t + n] == membership));
			d_crowd[n] = (float) 1.;
		}
	}
}

__global__ void set_segment_kernel(std::int32_t *d_segments, std::int16_t *d_memberships, const std::uint32_t N, const std::uint32_t t, std::int16_t crowd, std::int32_t segment) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		if (d_memberships[N*t + n] == crowd) {
			d_segments[n] = segment;
		}
	}
}

std::uint32_t identify_segments(SEGMENTS &h_segments, std::int16_t *h_memberships, float* h_memberships_apl, float* h_memberships_mpl, std::int8_t *h_spins, GRAPH &h_graph, const std::uint32_t N, const std::uint32_t T, const std::uint32_t mixing_time) {
	printf("Identifying memberships\n");

	//Compute the memberships at every time step.
	std::uint16_t *h_membership_count = new std::uint16_t[T]();
	std::fill(h_memberships, h_memberships + T*N, -1);
	membershipBFS(h_memberships, h_membership_count, h_memberships_apl, h_memberships_mpl, h_spins, h_graph, N, T, mixing_time);
	

	delete[] h_membership_count;

	printf("Identifying Segments\n");

	std::fill(h_segments.partitions, h_segments.partitions + N*T, -1);

	std::uint8_t *progression_check = new std::uint8_t[N*N]();

	size_t blocksize = 256;
	size_t blocknum = (N / blocksize) + 1;

	std::int16_t *d_memberships;
	cudaMalloc((void**)&d_memberships, sizeof(std::int16_t)*N*T);
	cudaMemcpy(d_memberships, h_memberships, sizeof(std::int16_t)*N*T, cudaMemcpyHostToDevice);

	std::int32_t *d_segments;
	cudaMalloc((void**)&d_segments, sizeof(std::int32_t)*N);

	//Set the segment number at t=0.
	std::int32_t segment = 0;
	std::int32_t curr_segment;
	std::int16_t crowd;

	const size_t t_start = mixing_time;

	for (size_t i = 0; i < N; i++) {
		h_segments.partitions[N*t_start + i] = h_memberships[N*t_start + i];
		if (h_memberships[N*t_start + i] > segment) {
			segment = h_memberships[N*t_start + i];
		}
	}
	
	std::int16_t curr_mbrid, next_mbrid;

	double similarity, curr_mag, next_mag;

	printf("Identifying Segments\n");

	for (size_t t = t_start; t < T; t++) {
		if (t % 200 == 0) {
			printf("t: %d / %d\n", t, T - 1);
		}


		//Reset the progression check matrix at the beginning of every time step.
		std::fill(progression_check, progression_check + N*N, 0);

		for (size_t n = 0; n < N; n++) {
			//Check every vertex at time t to see if it part of a segment.
			//If it is not part of a segment that vertex's crowd is made 
			//into a new segment.
			if (h_segments.partitions[N*t + n] < 0) {
				segment++;
				crowd = h_memberships[N*t + n];
				cudaMemcpy(d_segments, h_segments.partitions + N*t, sizeof(std::int32_t)*N, cudaMemcpyHostToDevice);
				set_segment_kernel<<<blocknum, blocksize>>>(d_segments, d_memberships, N, t, crowd, segment);
				cudaMemcpy(h_segments.partitions + N*t, d_segments, sizeof(std::int32_t)*N, cudaMemcpyDeviceToHost);
			}

			if (t < T - 1) {
				//Get the membership id of the current vertex now and in the next time step.
				curr_mbrid = h_memberships[N*t + n];
				next_mbrid = h_memberships[N*t + N + n];
				//Get the segment id of the current vertex.
				curr_segment = h_segments.partitions[N*t + n];

				//Check to see if the progression for the current membership id to the potential
				//next membership id has already been checked.
				if (progression_check[curr_mbrid*N + next_mbrid] == 0) {
					//Set the current potential progression as checked.
					progression_check[curr_mbrid*N + next_mbrid] = 1;

					//Set the similarity to 0.
					similarity = 0.;
					curr_mag = 0;
					next_mag = 0;
					for (size_t i = 0; i < N; i++) {
						//Compute the size of the current crowd.
						if (h_memberships[N*t + i] == curr_mbrid) {
							curr_mag++;
						}

						//Compute the size of the potential next crowd.
						if (h_memberships[N*t + N + i] == next_mbrid) {
							next_mag++;
						}

						//Compute the number of shared vertices in the current and potential crowds.
						if (h_memberships[N*t + i] == curr_mbrid && h_memberships[N*t + N + i] == next_mbrid) {
							similarity++;
						}
					}

					//Calculate the similarity of the crowds.
					similarity /= sqrt(curr_mag * next_mag);

					//If the similarity is greater than 85% then the potential crowd is a continuation of the current crowd.
					if (similarity >= 0.71) {
						//if (curr_mbrid != 0)
							//printf("\tS(%d, %d) = %f\n", curr_mbrid, next_mbrid, similarity);

						//printf("Similar!\n");
						cudaMemcpy(d_segments, h_segments.partitions + N*t + N, sizeof(std::int32_t)*N, cudaMemcpyHostToDevice);
						set_segment_kernel<<<blocknum, blocksize>>>(d_segments, d_memberships, N, t+1, next_mbrid, curr_segment);
						cudaMemcpy(h_segments.partitions + N*t + N, d_segments, sizeof(std::int32_t)*N, cudaMemcpyDeviceToHost);
					}
				}
			}
		}
	}

	delete[] progression_check;
	cudaFree(d_segments);
	cudaFree(d_memberships);

	return segment + 1;
}

void membership_similarity_thread(float *h_memsim, float *h_memsize, std::int16_t *h_memberships, std::int8_t *h_spins, const std::uint32_t N, const std::uint32_t T, const std::uint32_t t, const size_t thrid) {
	if (t < T - 1) {
		//Initialise the similarity matrix and member size array to zeros.
		std::fill(h_memsim, h_memsim + N*N, 0.f);
		std::fill(h_memsize, h_memsize + 2 * N, 0.f);

		size_t tN, tNp1;
		std::int16_t memt, memtp1;

		for (size_t i = 0; i < N; i++) {
			//store repeatedly calculated values in variables to slightly speed up things.
			tN = (t+thrid)*N;
			tNp1 = (t + thrid)*N + N;

			//For a membership x and a membership y add 1 to element xy in the similarty matrix.
			//Also count the size of each membership.
			if (h_spins[tN + i] == h_spins[tNp1 + i]) {
				memt = h_memberships[tN + i];
				memtp1 = h_memberships[tNp1 + i];
				h_memsim[memt*N + memtp1]++;
				h_memsize[memt]++;
				h_memsize[N + memtp1]++;
			}
		}
	}
}

__global__ void membership_similarity_normalise_kernel(float *d_memsim, float *d_memsize, const size_t N, const size_t R, const size_t C) {
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	int c = blockDim.y * blockIdx.y + threadIdx.y;

	if (r < R && c < C) {
		d_memsim[r*N+c]/= sqrtf(d_memsize[r] * d_memsize[N + c]);
	}
}

__global__ void membership_similarity_threshold_kernel(std::int16_t *d_bckmemlink, float *d_memsim, const size_t N, const size_t R, const size_t C, const size_t t) {
	int r = blockDim.x * blockIdx.x + threadIdx.x;

	if (r < R) {
		float maxsim = 0;
		std::int16_t maxsimid = 0;
		for (int c = 0; c < C; c++) {
			if (d_memsim[r*N + c] > maxsim) {
				maxsim = d_memsim[r*N + c];
				maxsimid = c;
			}
		}

		if (maxsim >= 0.72) {
			d_bckmemlink[t*N + N + maxsimid] = r;
		}
		else {
			d_bckmemlink[t*N + N + maxsimid] = -1;
		}
	}
}

__global__ void segment_progression_kernel(std::int32_t *d_membership_segmentids, std::int16_t *d_bckmemlink, std::uint16_t *d_memcount, std::int32_t *d_segid, const size_t N, const size_t T, const size_t t) {
	int m = blockDim.x * blockIdx.x + threadIdx.x;

	if (m < d_memcount[t]) {
		if (d_bckmemlink[N*t + m] < 0) {
			//If there is no backward membership link this is a start of a new segment.
			d_membership_segmentids[N*t + m] = atomicAdd(d_segid, 1);
		} else {
			//If the there is a backward membbership link this membership is a progression of a segment.
			std::int32_t memlink = d_bckmemlink[N*t + m];
			d_membership_segmentids[N*t + m] = d_membership_segmentids[N*t - N + memlink];
		}
	}
}

__global__ void segment_setter_kernel(std::int32_t *d_segment, std::int32_t *d_membership_segmentids, std::int16_t *d_memberships, std::uint16_t *d_memberships_count, const size_t N, const size_t T, const size_t analysis_time) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	int t = blockDim.y * blockIdx.y + threadIdx.y;

	if (n < N && t < T && t >= analysis_time) {
		int memid = d_memberships[N*t + n];
		if (memid >= 0 && memid < d_memberships_count[t]) {
			d_segment[N*t + n] = d_membership_segmentids[N*t + memid];
		}
	}
}

std::uint32_t identify_segments2(SEGMENTS &h_segments, std::int16_t *h_memberships, float* h_memberships_apl, float* h_memberships_mpl, std::int8_t *d_spins, std::int8_t *h_spins, GRAPH &h_graph, const std::uint32_t N, const std::uint32_t T, const std::uint32_t analysis_time) {
	//The arrays required to compute the segment progressions.
	std::int16_t *d_memberships;				//The membership partitions for each time step.
	std::uint16_t *d_membership_count;			//The number of memberships at time t.
	std::uint16_t *d_deglist;					//Graph degree list.
	std::uint16_t *d_adjlist;					//Graph adjacency list.

	size_t K = h_graph.max_deg;	//The maximum degree in the graph.

	cudaMalloc((void**)&d_memberships, sizeof(std::int16_t)*N*T);
	cudaMalloc((void**)&d_membership_count, sizeof(std::uint16_t)*T);

	cudaMalloc((void**)&d_deglist, sizeof(std::uint16_t)*N);
	cudaMalloc((void**)&d_adjlist, sizeof(std::uint16_t)*N*K);

	cudaMemcpy(d_deglist, h_graph.deglist, sizeof(std::uint16_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjlist, h_graph.adjlist, sizeof(std::uint16_t)*N*K, cudaMemcpyHostToDevice);

	//Compute the memberships at every time step.
	std::uint16_t *h_membership_counts = new std::uint16_t[T]();
	cudaDeviceSynchronize();
	compute_membership_device(d_memberships, d_membership_count, d_spins, d_deglist, d_adjlist, N, K, T, analysis_time);
	cudaDeviceSynchronize();
	cudaMemcpy(h_memberships, d_memberships, sizeof(std::int16_t)*N*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_membership_counts, d_membership_count, sizeof(std::uint16_t)*T, cudaMemcpyDeviceToHost);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	const size_t numOfThreads = 7;
	std::thread *threads = new std::thread[numOfThreads];

	float **h_memsims = new float*[numOfThreads];	//The membership similarity matrices for each thread.	
	float **h_memsizes = new float*[numOfThreads];	//The sizes of each membership for each thread.

	//Create the similarity matrices and membership size arrays.
	for (size_t thr = 0; thr < numOfThreads; thr++) {
		h_memsims[thr] = new float[N*N]();
		h_memsizes[thr] = new float[2 * N]();
	}

	//The device version of the membership similarity and size arrays.
	float *d_memsim, *d_memsize;
	cudaMalloc((void**)&d_memsim, sizeof(float)*N*N);
	cudaMalloc((void**)&d_memsize, sizeof(float)*2*N);


	std::int16_t *d_bckmemlink;					//The membership at time t-1 to which a membership at time t is linked.
	std::int32_t *d_membership_segmentids;		//The segment id assigned to each membership during each time step.
	std::int32_t *d_segments;					//The segment id assigned to each vertex during each time step.
	std::int32_t *d_segid;						//A counter used to assign segment ids to memberships.

	cudaMalloc((void**)&d_bckmemlink, sizeof(std::int16_t)*N*T);
	cudaMalloc((void**)&d_membership_segmentids, sizeof(std::int32_t)*N*T);
	cudaMalloc((void**)&d_segments, sizeof(std::int32_t)*N*T);
	cudaMalloc((void**)&d_segid, sizeof(std::int32_t));

	cudaMemset(d_bckmemlink, -1, sizeof(std::int16_t)*N*T);
	cudaMemset(d_segid, 0, sizeof(std::int32_t));
	cudaMemset(d_membership_segmentids, -1, sizeof(std::int32_t)*N*T);

	//The block number and size for the 
	dim3 blocknum, blocksize;
	blocksize.x = blocksize.y = 32;
	blocknum.x = (N / blocksize.x) + 1;
	blocknum.y = (N / blocksize.y) + 1;	

	size_t R, C;

	/*
	std::fill(h_memberships, h_memberships + T*N, -1);
	membershipBFS(h_memberships, h_membership_count, h_memberships_apl, h_memberships_mpl, h_spins, h_graph, N, T);
	cudaMemcpy(d_memberships, h_memberships, sizeof(std::int16_t)*N*T, cudaMemcpyHostToDevice);
	cudaMemcpy(d_membership_count, h_membership_count, sizeof(std::uint16_t)*T, cudaMemcpyHostToDevice);
	*/

	printf("\t\tIdentifying segments\n");
	for (size_t t = analysis_time; t < T - numOfThreads; t += numOfThreads) {
		/*
		if (t % (numOfThreads * 10) == 0) {
			printf("t: %d / %d\n", t, T);
		}
		*/

		//Create threads to compute the similarity between memberships at time t and t+1.
		for (size_t thr = 0; thr < numOfThreads; thr++) {
			threads[thr] = std::thread(membership_similarity_thread, h_memsims[thr], h_memsizes[thr], h_memberships, h_spins, N, T, t, thr);
		}

		for (size_t thr = 0; thr < numOfThreads; thr++) {
			//Join the threads.
			threads[thr].join();

			//Copy the similarity matrix for t+thr
			cudaMemcpy(d_memsim, h_memsims[thr], sizeof(float)*N*N, cudaMemcpyHostToDevice);
			cudaMemcpy(d_memsize, h_memsims[thr], sizeof(float) * 2 * N, cudaMemcpyHostToDevice);
			
			//Normalise the similarity matrix.
			R = h_membership_counts[t + thr];
			C = h_membership_counts[t + thr + 1];
			membership_similarity_normalise_kernel<<<blocknum, blocksize>>>(d_memsim, d_memsize, N, R, C);
			cudaDeviceSynchronize();
			//Compute the backward membership link matrix.
			membership_similarity_threshold_kernel<<<(R / 256) + 1, 256 >>>(d_bckmemlink, d_memsim, N, R, C, t + thr);
			cudaDeviceSynchronize();
		}
	}

	//Compute the membership segment ids.
	for (size_t t = analysis_time; t < T; t++) {
		segment_progression_kernel <<< (N / 256) + 1, 256 >>>(d_membership_segmentids, d_bckmemlink, d_membership_count, d_segid, N, T, t);
		cudaDeviceSynchronize();
	}

	//Set the node segment ids.
	blocksize.x = blocksize.y = 32;
	blocknum.x = (N / blocksize.x) + 1;
	blocknum.y = (T / blocksize.y) + 1;
	segment_setter_kernel<<< blocknum, blocksize >>>(d_segments, d_membership_segmentids, d_memberships, d_membership_count, N, T, analysis_time);
	cudaDeviceSynchronize();

	cudaMemcpy(h_segments.partitions, d_segments, sizeof(std::int32_t)*N*T, cudaMemcpyDeviceToHost);

	std::int32_t numOfSegments = 0;
	cudaMemcpy(&numOfSegments, d_segid, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//Free all memory.
	cudaFree(d_memsim);
	cudaFree(d_memsize);

	cudaFree(d_memberships);
	cudaFree(d_segments);
	cudaFree(d_bckmemlink);
	cudaFree(d_segid);
	cudaFree(d_membership_segmentids);
	cudaFree(d_membership_count);
	cudaFree(d_deglist);
	cudaFree(d_adjlist);

	for (size_t thr = 0; thr < numOfThreads; thr++) {
		delete[] h_memsims[thr];
		delete[] h_memsizes[thr];
	}

	delete[] h_membership_counts;
	delete[] threads;
	delete[] h_memsims;
	delete[] h_memsizes;

	return numOfSegments;
}

void compute_lifetimes(SEGMENTS &h_segments, const std::uint32_t mixing_time) {
	std::fill(h_segments.birthdeath, h_segments.birthdeath + 2 * h_segments.S, -1);

	const size_t init_t = mixing_time;
	std::int32_t seg;

	for (size_t t = init_t; t < h_segments.T; t++) {
		size_t Nt = h_segments.N * t;
		for (size_t n = 0; n < h_segments.N; n++) {
			seg = h_segments.partitions[Nt + n];

			if (seg >= 0) {
				if (seg > h_segments.S) {
					printf("\tseg = %d\n", seg);
				}
				if (h_segments.birthdeath[2 * seg] < 0) {
					h_segments.birthdeath[2 * seg] = (std::uint32_t) t;
				}
				h_segments.birthdeath[2 * seg + 1] = (std::uint32_t) t;
			}
		}
	}

	for (seg = 0; seg < h_segments.S; seg++) {
		h_segments.lifetimes[seg] = 1 + h_segments.birthdeath[2 * seg + 1] - h_segments.birthdeath[2 * seg];
	}
}

__global__ void divide_apl_kernel(float *d_segment_apl, std::uint16_t *d_segment_pop, std::uint16_t *d_birthdeath, const size_t T) {
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t < T) {
	}
}

void compute_segment_centres_thread(SEGMENTS *h_segments, float *h_memberships_apl, float *h_memberships_mpl, std::int32_t *g_apsp, size_t N, const size_t t) {
	std::int32_t seg;
	size_t relt;

	for (size_t n = 0; n < N; n++) {
		seg = h_segments->partitions[N*t + n];
		relt = t - h_segments->birthdeath[2 * seg];
		if (relt == 0) {
			if (seg < 0 || seg >= h_segments->S || relt >= h_segments->lifetimes[seg]) {
				printf("(%d, %d) : [%d / %d, %d / %d]\n", t, n, seg, h_segments->S, relt, h_segments->ML);
			}

			if (h_memberships_apl[N*t + n] < h_segments->apl_min[seg][relt]) {
				h_segments->centre[seg] = n;
			}
		}
	}
}

void compute_pathlengths_thread(SEGMENTS *h_segments, float *h_memberships_apl, float *h_memberships_mpl, std::int32_t *g_apsp, size_t N, const size_t t) {
	std::int32_t seg;
	size_t relt;

	for (size_t n = 0; n < N; n++) {
		seg = h_segments->partitions[N*t + n];
		relt = t - h_segments->birthdeath[2 * seg];

		if (seg < 0 || seg >= h_segments->S || relt >= h_segments->lifetimes[seg]) {
			printf("(%d, %d) : [%d / %d, %d / %d]\n", t, n, seg, h_segments->S, relt, h_segments->ML);
		}

		h_segments->pop[seg][relt]++;
		h_segments->apl[seg][relt] += h_memberships_apl[N*t + n];

			
		if (h_memberships_apl[N*t + n] < h_segments->apl_min[seg][relt]) {
			//Set the min pathlength.
			h_segments->apl_min[seg][relt] = h_memberships_apl[N*t + n];

			//Distance from the starting vertex
			size_t tmpind = h_segments->centre[seg] * N + n;
			h_segments->dist_from_start[seg][relt] = (float)g_apsp[tmpind];
		}

		h_segments->apl_max[seg][relt] = std::max(h_memberships_apl[N*t + n], h_segments->apl_max[seg][relt]);
		h_segments->mpl[seg][relt] = std::max(h_memberships_mpl[N*t + n], h_segments->mpl[seg][relt]);
	}
}

void compute_pathlengths(SEGMENTS &h_segments, float *h_memberships_apl, float *h_memberships_mpl, std::int32_t *g_apsp, const size_t analysis_time) {
	printf("\t\t[%d, %d, %d, %d]\n", h_segments.N, h_segments.T, h_segments.S, h_segments.ML);

	if (h_segments.S > 0) {
		std::int32_t seg;
		std::uint32_t relt;

		const size_t N = h_segments.N;

		const size_t numOfThreads = 7;
		std::thread threads[numOfThreads];

		// Find the first centre vertex of every segment.
		for (size_t t = analysis_time; t < h_segments.T; t += numOfThreads) {
			for (size_t thr = 0; thr < numOfThreads; thr++) {
				if (t + thr < h_segments.T) {
					threads[thr] = std::thread(compute_segment_centres_thread, &h_segments, h_memberships_apl, h_memberships_mpl, g_apsp, N, t + thr);
				}
			}
			for (size_t thr = 0; thr < numOfThreads; thr++) {
				if (t + thr < h_segments.T) {
					if (threads[thr].joinable()) {
						threads[thr].join();
					}
				}
			}
		}

		// Calculate the average, shortest average, longest average and maximum pathlength of every segment for every timestep it exists.
		for (size_t t = analysis_time; t < h_segments.T; t += numOfThreads) {
			for (size_t thr = 0; thr < numOfThreads; thr++) {
				if (t + thr < h_segments.T) {
					threads[thr] = std::thread(compute_pathlengths_thread, &h_segments, h_memberships_apl, h_memberships_mpl, g_apsp, N, t + thr);
				}
			}
			for (size_t thr = 0; thr < numOfThreads; thr++) {
				if (t + thr < h_segments.T) {
					if (threads[thr].joinable()) {
						threads[thr].join();
					}
				}
			}
		}


		// Divide the average pathlength of the segments by the population at each timestep they exists.
		for (size_t s = 0; s < h_segments.S; s++) {
			for (size_t l = 0; l < h_segments.lifetimes[s]; l++) {
				h_segments.apl[s][l] /= (float) h_segments.pop[s][l];
			}
		}


		//Calculate the variance of the average pathlengths of the segments for each timestep they exists.
		for (size_t t = analysis_time; t < h_segments.T; t++) {
			for (size_t n = 0; n < N; n++) {
				seg = h_segments.partitions[N*t + n];
				relt = (std::uint32_t) t - h_segments.birthdeath[2 * seg];
				h_segments.vpl[seg][relt] += (float) std::pow(h_memberships_apl[N*t + n] - h_segments.apl[seg][relt], 2.);
			}
		}


		// Divide the variance of the average pathlengths of the segments by the population for each timestep they exists.
		for (size_t s = 0; s < h_segments.S; s++) {
			for (size_t l = 0; l < h_segments.lifetimes[s]; l++) {
				h_segments.vpl[s][l] /= (float)h_segments.pop[s][l];
			}
		}
	}
}

/*
__device__ int nchoosek(const float n, const float k) {
	return tgamma(n + 1) / (tgamma(n - k + 1.f)*tgamma(k + 1.f));
}

__global__ void segment_property1_kernel(float *d_intdegs, float *d_extdegs, float *d_cc, std::int32_t *d_segments, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, std::uint8_t *d_adjmat, size_t N, size_t K, size_t T, size_t mix_time) {
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t >= mix_time && t < T) {
		int Nt = N*t;
		int nK;
		int m, mm, s, Nm;
		float numOfTriangles;

		for (int n = 0; n < N; n++) {
			s = d_segments[Nt + n];
			numOfTriangles = 0;
			nK = n*K;

			for (int k = 0; k < d_deglist[n]; k++) {
				m = d_adjlist[nK + k]; //Get a neighbour m.

				if (d_segments[Nt + m] == s) {
					d_intdegs[Nt + n]++;	//Count the number internal degrees of n.

					//Count the number of triangles.
					for (int kk = 0; kk < d_deglist[n]; kk++) {
						mm = d_adjlist[nK + kk];	//Get neighbour mm.
						//Check if m and mm are connected and in the same segment.
						if (d_segments[Nt + mm] == s) {
							if (d_adjmat[N*m + mm] == 1) {
								numOfTriangles++;
							}
						}
					}
				}
				else {
					//Count the number of external degrees.
					d_extdegs[Nt + n]++;
				}
			}

			float numOfTriplets = (float) nchoosek(d_intdegs[Nt + n], 2.f);
			d_cc[Nt + n] = numOfTriangles / (2.f * numOfTriplets);
		}
	}
}

void segment_properties_thread(SEGMENTS *h_segments, std::uint8_t *segvisited, std::vector<size_t> *segs, float *h_intdegs, float *h_extdegs, float *h_ccs, const size_t N, const float totK, const size_t t) {
	size_t s, l, Nt(N*t);

	//Sum the internal degrees, the external degrees and the 
	//internal clustering coefficients for each segment.
	for (size_t n = 0; n < N; n++) {
		s = h_segments->partitions[N*t + n];
		l = t - h_segments->birthdeath[2 * s];

		if (segvisited[s] < 1) {
			segvisited[s] = 1;
			segs->push_back(s);
		}

		h_segments->mods[s][l] += h_intdegs[Nt + n];
		h_segments->extdegs[s][l] += h_extdegs[Nt + n];
		h_segments->cc_means[s][l] += h_ccs[Nt + n];
	}

	//Compute the modularities and mean internal clustering coefficients of the segments.
	for (size_t i = 0; i < segs->size(); i++) {
		s = (*segs)[i];
		l = t - h_segments->birthdeath[2 * s];

		h_segments->mods[s][l] /= totK;
		h_segments->extdegs[s][l] /= totK;
		h_segments->mods[s][l] -= h_segments->extdegs[s][l] * h_segments->extdegs[s][l];
		h_segments->cc_means[s][l] /= (float) h_segments->pop[s][l];
	}

	//Compute the internal clustering coefficient variance.
	for (size_t n = 0; n < N; n++) {
		s = h_segments->partitions[N*t + n];
		l = t - h_segments->birthdeath[2 * s];

		h_segments->cc_vars[s][l] += std::pow(h_ccs[Nt + n] - h_segments->cc_means[s][l], 2.);
	}

	for (size_t i = 0; i < segs->size(); i++) {
		s = (*segs)[i];
		l = t - h_segments->birthdeath[2 * s];

		h_segments->cc_vars[s][l] /= (float) h_segments->pop[s][l];
	}
}

void compute_segment_properties(SEGMENTS &h_segments, GRAPH &h_graph, const size_t mix_time, const size_t relax_time, const size_t T) {
	//Compute the modularity and the internal clustering coefficients for each segment.

		On Device:
		1. Compute the internal and external degrees for each node for each time t.
		2. Compute the internal clustering coefficient for each node for each time t.
		
		1 and 2 can be done together.

		On Host:
		3. For each time t compute the modularities of the segments using the int and ext degrees in 1.
		4. For each time t compute the internal clustering coefficients of the segments using the clustering coefficients in 2.

		3 and 4 can be done at the same time. 

	const size_t N = h_segments.N;
	const size_t T = h_segments.T;
	const size_t K = h_graph.max_deg;

	std::int32_t *d_partitions;
	float *d_intdegs, *d_extdegs, *d_nodeccs;
	std::uint16_t *d_deglist, *d_adjlist;
	std::uint8_t *d_adjmat;

	//Copy arrays to device that are needed to compute the correlation coefficients and modularities.
	cudaMalloc((void**)&d_partitions, sizeof(std::int32_t)*N*T);
	cudaMalloc((void**)&d_deglist, sizeof(std::uint16_t)*N);
	cudaMalloc((void**)&d_adjlist, sizeof(std::uint16_t)*N*K);
	cudaMalloc((void**)&d_adjmat, sizeof(std::uint8_t)*N*N);

	cudaMemcpy(d_partitions, h_segments.partitions, sizeof(std::int32_t)*N*T, cudaMemcpyHostToDevice);
	cudaMemcpy(d_deglist, h_graph.deglist, sizeof(std::uint16_t)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjlist, h_graph.adjlist, sizeof(std::uint16_t)*N*K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjmat, h_graph.adjmat, sizeof(std::uint8_t)*N*N, cudaMemcpyHostToDevice);
	 
	//Create arrays on device to store results of kernel.
	cudaMalloc((void**)&d_intdegs, sizeof(float)*N*T);
	cudaMalloc((void**)&d_extdegs, sizeof(float)*N*T);
	cudaMalloc((void**)&d_nodeccs, sizeof(float)*N*T);

	//Run the kernel.
	size_t blocksize = 256;
	size_t blocknum = (T / blocksize) + 1;
	segment_property1_kernel<<<blocknum, blocksize>>>(d_intdegs, d_extdegs, d_nodeccs, d_partitions, d_deglist, d_adjlist, d_adjmat, N, K, T, mix_time);

	//Copy the results to the host.
	float *h_intdegs = new float[N*T]();
	float *h_extdegs = new float[N*T]();
	float *h_nodeccs = new float[N*T]();

	cudaMemcpy(h_intdegs, d_intdegs, sizeof(float)*N*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_extdegs, d_extdegs, sizeof(float)*N*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_nodeccs, d_nodeccs, sizeof(float)*N*T, cudaMemcpyDeviceToHost);

	//Create the arrays needed for the threads.
	const size_t numOfThreads = 7;
	std::thread threads[numOfThreads];

	std::uint8_t *segvisited = new std::uint8_t[numOfThreads*h_segments.S];
	std::vector<std::uint16_t> segs[numOfThreads];

	//calculate the total degree of the graph.
	double totK = 0;
	for (size_t i = 0; i < N; i++) {
		totK += h_graph.deglist[i];
	}

	for (size_t t = mix_time; t < T; t += numOfThreads) {
		for (size_t thr = 0; thr < numOfThreads; thr++) {
			threads[thr] = std::thread(h_segments, segvisited + thr*h_segments.S, segs[thr], h_intdegs, h_extdegs, h_nodeccs, N, totK, t);
		}

		for (size_t thr = 0; thr < numOfThreads; thr++) {
			threads[thr].join();
		}
	}

	delete[] h_intdegs;
	delete[] h_extdegs;
	delete[] h_nodeccs;
	delete[] segvisited;

	cudaFree(d_partitions);
	cudaFree(d_intdegs);
	cudaFree(d_extdegs);
	cudaFree(d_nodeccs);
	cudaFree(d_deglist);
	cudaFree(d_adjlist);
	cudaFree(d_adjmat);
}
*/

__global__ void partition_countmat_kernel(float *d_comsims, float *d_countmat, float *d_commat, float *d_commag, std::int32_t *d_partitions, size_t N, size_t t, size_t window) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		//Precalculate some repeated calculations.
		int Nn = N*n;
		int tpw = t + window;
		int tmwm1 = t - window - 1;
		int Ntpw = N * tpw;
		int Ntmwm1 = N * tmwm1;

		//Get the membership number of node n at t-window-1 and t+window.
		const int membrem = d_partitions[Ntmwm1+n];
		const int membadd = d_partitions[Ntpw+n];

		//Set the magnitude and simularity to n.
		double sum = 0;
		double sim = 0;

		for (int m = 0; m < N; m++) {
			//Remove the additions due to the times outside of the window.
			if (membrem == d_partitions[Ntmwm1 + m]) {
				d_countmat[Nn + m]--;
			}

			//Add the additions due to the times now inside the window.
			if (membadd == d_partitions[Ntpw + m]) {
				d_countmat[Nn + m]++;
			}

			//Compute the magnitude.
			sum += d_countmat[Nn + m];
			//Compute the product.
			sim += d_countmat[Nn + m] * d_commat[Nn + m];
		}

		//Compute the similarity
		d_comsims[N*t + n] = sim / sqrt(d_commag[n] * sum);
	}
}

__global__ void community_consensus_kernel(std::int16_t *d_consensus, std::int8_t *d_spins, std::uint16_t *d_comms, size_t N,  size_t C, size_t T, size_t mix_time) {
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t >= mix_time && t < T) {
		int com;
		int Nt = N*t;
		for (size_t n = 0; n < N; n++) {
			com = d_comms[n];
			d_consensus[com*T + t] += d_spins[Nt + n];
		}
	}
}


void compute_community_properties(COMMUNITY_DATA &comm, SEGMENTS &h_segments, std::int8_t *d_spins, const size_t N, const size_t T, const size_t mix_time) {
	if (comm.loaded) {
		std::int16_t *d_consensus;
		cudaMalloc((void**)&d_consensus, sizeof(std::int16_t)*comm.C*T);
		cudaMemset(d_consensus, 0, sizeof(std::int16_t)*comm.C*T);

		std::uint16_t *d_comms;
		cudaMalloc((void**)&d_comms, sizeof(std::uint16_t)*N);
		cudaMemcpy(d_comms, comm.memberships, sizeof(std::uint16_t)*N, cudaMemcpyHostToDevice);

		size_t blocksize = 128;
		size_t blocknum = (T / blocksize) + 1;

		community_consensus_kernel << <blocknum, blocksize >> >(d_consensus, d_spins, d_comms, N, comm.C, T, mix_time);

		cudaMemcpy(comm.consensus, d_consensus, sizeof(std::int16_t)*comm.C*T, cudaMemcpyDeviceToHost);

		cudaFree(d_consensus);
		cudaFree(d_comms);
	}
}


__global__ void compute_membership_kernel(std::int16_t *d_memberships, std::uint16_t *d_memberships_count, std::uint16_t *d_qmat, std::int8_t *d_spins, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, size_t N, size_t K, size_t T, size_t analysis_time) {
	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if(t >= analysis_time && t < T) {
		int Nt = N*t;
		int memid = 0;	//membership id
		int qfront = 0;	//Current front element of queue.
		int qback = 0;	//Current insertion point of queue.
		int x, y, s;	//node id and spin holders.

		for (int n = 0; n < N; n++) {
			//Check if the node is already in a crowd.
			if (d_memberships[Nt + n] < 0) {
				s = d_spins[Nt + n];			//Get its spin.
				if (qback < N) {
					d_qmat[Nt + qback] = n;			//Add it to the back of the queue.
					qback++;
				}					//Move the back of the queue on step.
				d_memberships[Nt + n] = memid;	//Set the node's membership id.

				while (qfront < qback) {
					x = d_qmat[Nt + qfront];	//Get the node at the front of the queue.
					qfront++;					//Move the front of the queue back one step.
					//Check the neighbours of node x.
					for (int k = 0; k < d_deglist[x]; k++) {
						y = (int) d_adjlist[x*K + k];
						//Check if node y has already been assigned a membership id.
						if (d_memberships[Nt + y] < 0) {
							d_memberships[Nt + y] = memid;
							//Check if node y has the same spin as x.
							if (d_spins[Nt + y] == s) {
									d_qmat[Nt + qback] = y;			//Add node y to the queue.
									qback++;
								//d_memberships[Nt + y] = memid;	//Assign node y a membership id
							}
						}
					}
				}
				memid++;
			}
		}
		
		d_memberships_count[t] = memid;
	}
}

void compute_membership_device(std::int16_t *d_memberships, std::uint16_t *d_memberships_count, std::int8_t *d_spins, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, const size_t N, const size_t K, const size_t T, const size_t analysis_time) {
	std::uint16_t *d_qmat;
	cudaMalloc((void**)&d_qmat, sizeof(std::uint16_t)*N*T);

	cudaMemset(d_memberships, (std::int16_t) -1, sizeof(std::int16_t)*N*T);
	cudaMemset(d_qmat, 0, sizeof(std::uint16_t)*N*T);

	cudaDeviceSynchronize();

	const size_t blocksize = 128;
	const size_t blocknum = (T / blocksize) + 1;

	printf("\tStarting Membership Kernel: %d, %d\n", analysis_time, T);
	compute_membership_kernel<<<blocknum, blocksize>>>(d_memberships, d_memberships_count, d_qmat, d_spins, d_deglist, d_adjlist, N, K, T, analysis_time);
	printf("\tMembership Kernel Finished\n");

	cudaFree(d_qmat);
}

void edgecounter_thread(SEGMENTS *h_segments, GRAPH *h_graph, const size_t t_offset, const size_t numOfThreads) {
	//The time steps are split amoung several threads. 
	//Each thread will continue until it has counted the edges for all of its assigned time steps.
	for (size_t t = t_offset; t < (*h_segments).T; t += numOfThreads) {
		for (size_t n = 0; n < (*h_graph).NoV; n++) {

			//Get the crowd of vertex n during time step t.
			std::int32_t s = (*h_segments).partitions[(*h_segments).N * t + n];

			//Get the relative time of the crowd.
			long int rt = t - (*h_segments).birthdeath[2 * s];

			(*h_segments).extdegs[s][rt] += (*h_graph).deglist[n];

			//Check the neighbours of vertex n to see if they are in crowd s or not.
			for (size_t k = 0; k < (*h_graph).deglist[n]; k++) {
				//Get the neighbour vertex id.
				size_t m = (*h_graph).adjlist[n*(*h_graph).max_deg + k];
				//Get the crowd of vertex m.
				std::uint32_t sm = (*h_segments).partitions[(*h_segments).N * t + m];

				if (s == sm) {
					//If vertex n and m are in the same crowd increment the number of internal edges
					//during time step rt.
					(*h_segments).intdegs[s][rt]++;
				}
			}
		}
	}
}

void modularity_thread(SEGMENTS *h_segments, const double numOfDegrees, const size_t s_offset, const size_t numOfThreads) {
	for (size_t s = s_offset; s < (*h_segments).S; s += numOfThreads) {
		for (size_t l = 0; l < (*h_segments).lifetimes[s]; l++) {
			double ai = (double)(*h_segments).extdegs[s][l] / (double)numOfDegrees / 2.;
			(*h_segments).mods[s][l] = (float)((double)(*h_segments).intdegs[s][l] / (double)numOfDegrees / 2. - ai*ai);
		}
	}
}

void compute_modularities(SEGMENTS &h_segments, GRAPH &h_graph, const size_t mixing_time) {
	/*
		- Count number of internal and external degrees for each crowd during each time step.
		- intedges - extedges^2 is the modularity of the crowd.
		- Can do this parallel on host using threads.

		Q = e_{ii} - a_i^2;

		e_{ij} = \sum_{vw} A_{vw} \delta\left(v\in C_i\right)\delta\left(w\in C_j\right) / (2m);

		a_i = \sum_j e_{ij};
		*/

	//Count the number of internal edges and the total number of edges in each crowd during each time step of its life.
	const size_t numOfThreads = 7;

	std::thread edgecounter_thrds[numOfThreads];
	for (size_t thrd = 0; thrd < numOfThreads; thrd++) {
		edgecounter_thrds[thrd] = std::thread(edgecounter_thread, &h_segments, &h_graph, mixing_time + thrd, numOfThreads);
	}

	for (size_t thrd = 0; thrd < numOfThreads; thrd++) {
		edgecounter_thrds[thrd].join();
	}

	//Count the total number of degrees in the graph.
	double totaldeg = 0;
	for (size_t i = 0; i < h_graph.NoV; i++) {
		totaldeg += h_graph.deglist[i];
	}

	std::thread mod_thrds[numOfThreads];
	for (size_t thrd = 0; thrd < numOfThreads; thrd++) {
		mod_thrds[thrd] = std::thread(modularity_thread, &h_segments, totaldeg, thrd, numOfThreads);
	}

	for (size_t thrd = 0; thrd < numOfThreads; thrd++) {
		mod_thrds[thrd].join();
	}
}

__global__ void community_associationinit_kernel(float *d_comm_association, float *d_comm_assmag_sq, std::uint16_t* d_communities, const size_t N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		d_comm_assmag_sq[n] = 0;
		const int nc = d_communities[n];
		for (int m = 0; m < N; m++) {
			const int mc = d_communities[m];

			d_comm_association[n*N + m] = (float) 0;
			if (mc == nc) {
				//printf(" (%d,%d)", n, m);
				d_comm_association[n*N + m] = (float) 1;
				d_comm_assmag_sq[n] += (float) 1;
			}
		}
	}
}

__global__ void crowd_associationinit_kernel(float *d_crowd_association, float *d_crowd_assmag_sq, std::int32_t *d_partition, const size_t N, const size_t mixing_time, const size_t W) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		//Count the numebr of time each vertex m is in the same crowd as vertex n
		//during the time period [0,2W+1].
		const int Win = 2 * W + 1;
		for (int t = mixing_time; t < mixing_time+Win; t++) {
			int ns = d_partition[N*t + n];

			for (int m = 0; m < N; m++) {
				int ms = d_partition[N*t + m];

				if (ns == ms) {
					//Add an association between vertices n and m.
					d_crowd_association[N*n + m] += (float) 1;
				}
			}
		}

		//Compute the magnitude squared of the n-th row of the d_crowd_association matrix.
		d_crowd_assmag_sq[n] = 0;
		for (int m = 0; m < N; m++) {
			d_crowd_assmag_sq[n] += pow(d_crowd_association[N*n + m], 2.f);
		}
	}
}

__global__ void crowd_association_kernel(float *d_crowd_association, float *d_crowd_assmag_sq, std::int32_t *d_partition, const size_t t, const size_t N, const int addsub) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < N) {
		int ns = d_partition[N*t + n];
		for (int m = 0; m < N; m++) {
			int ms = d_partition[N*t + m];
			if (ns == ms) {
				//Remove the previous contribution to the magnitude squared of the nm-th association.
				d_crowd_assmag_sq[n] -= (float) pow(d_crowd_association[N*n + m], 2.f);
				//Change the association between vertices n and m.
				d_crowd_association[N*n + m] += (float) addsub;
				//Include the new contribution to the magnitude squared of the nm-th association.
				d_crowd_assmag_sq[n] += (float) pow(d_crowd_association[N*n + m], 2.f);
			}
		}
	}
}

__global__ void comm_simmeasure_kernel(float *d_comm_cossim, float *d_comm_eucdis, float *d_comm_association, float *d_comm_assmag_sq, float *d_crowd_association, float *d_crowd_assmag_sq, const size_t N, const size_t t, const size_t W) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		
		float totsim = 0;
		float totdis = 0;
		float WSz = 2.f * (float) W + 1.f;

		for (int m = 0; m < N; m++) {
			//Computes the cosine simularity between rows of d_comm_association and d_crowd_association.
			totsim += (float) d_comm_association[N*n + m] * d_crowd_association[N*n + m];
			//Compute the Euclidian distance between rows of d_comm_association and d_crowd_association.
			totdis += (float) pow(d_comm_association[N*n + m] - d_crowd_association[N*n + m] / WSz, 2.f);
		}
	
		//Divide the similarity by the magnitudes of the rows.
		totsim /= (float) sqrt(d_comm_assmag_sq[n] * d_crowd_assmag_sq[n]);
		d_comm_cossim[N*t + n] = totsim;

		//Divide distance by the maximum possible distance to scale the result to [0,1].
		totdis = (float) sqrt(totdis);
		d_comm_eucdis[N*t + n] = totdis;
	}
}


void compute_association(SEGMENTS &h_segments, const size_t analysis_time, const size_t W) {
	std::printf("Computing Community Associations\n");

	const size_t N = h_segments.N;
	const size_t T = h_segments.T;

	//Initialise memory on device.
	//Will store the cosine similarities and euclidian distances for each vertex at each time step.
	float *d_comm_cossim, *d_comm_eucdis;
	cudaMalloc((void**)&d_comm_cossim, sizeof(float)*N*T);
	cudaMalloc((void**)&d_comm_eucdis, sizeof(float)*N*T);

	cudaMemset(d_comm_cossim, 0, sizeof(float)*N*T);
	cudaMemset(d_comm_eucdis, 0, sizeof(float)*N*T);

	//The community vector.
	std::uint16_t *d_communities;
	cudaMalloc((void**)&d_communities, sizeof(std::uint16_t)*N);
	cudaMemcpy(d_communities, h_segments.communities, sizeof(std::uint16_t)*N, cudaMemcpyHostToDevice);

	//The community association matrix and magnitudes for each vertex.
	float *d_comm_association, *d_comm_assmag_sq;
	cudaMalloc((void**)&d_comm_association, sizeof(float)*N*N);
	cudaMalloc((void**)&d_comm_assmag_sq, sizeof(float)*N);

	//The crowd association matrix and magnitudes for each vertex.
	float *d_crowd_association, *d_crowd_assmag_sq;
	cudaMalloc((void**)&d_crowd_association, sizeof(float)*N*N);
	cudaMalloc((void**)&d_crowd_assmag_sq, sizeof(float)*N);

	//Copy the the partition matrix to the device.
	std::int32_t *d_partition;
	cudaMalloc((void**)&d_partition, sizeof(float)*N*T);
	cudaMemcpy(d_partition, h_segments.partitions, sizeof(std::int32_t)*N*T, cudaMemcpyHostToDevice);

	/*------------------------------------------------------------------------------------------------------*/

	const size_t threadNum = 64;

	//Initialise the community association values;
	community_associationinit_kernel<<<(N/threadNum)+1, threadNum>>>(d_comm_association, d_comm_assmag_sq, d_communities, N);

	//Initialise the initial crowd association values.
	crowd_associationinit_kernel<<<(N / threadNum) + 1, threadNum>>>(d_crowd_association, d_crowd_assmag_sq, d_partition, N, analysis_time, W);

	//Compute the similarity and the distance for each vertex.
	comm_simmeasure_kernel<<<(N/threadNum)+1, threadNum>>>(d_comm_cossim, d_comm_eucdis, d_comm_association, d_comm_assmag_sq, d_crowd_association, d_crowd_assmag_sq, N, analysis_time+W, W);

	float *h_crowd_association = new float[N*N]();
	float *h_cossim = new float[N]();
	float *h_eucdis = new float[N]();
	cudaMemcpy(h_crowd_association, d_crowd_association, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cossim, d_comm_cossim+analysis_time*N, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eucdis, d_comm_eucdis+analysis_time*N, sizeof(float)*N, cudaMemcpyDeviceToHost);

	printf("Similarity = %f\n", h_cossim[0]);

	delete[] h_crowd_association;
	delete[] h_cossim;
	delete[] h_eucdis;

	//Remove the contributions of the first time step used for analysis
	crowd_association_kernel<<<(N/threadNum)+1, threadNum>>>(d_crowd_association, d_crowd_assmag_sq, d_partition, analysis_time, N, -1);

	//Compute the similarities and distances for all other time steps in the interval (W,T-W).
	for (size_t t = analysis_time + W + 1; t < T - W; t++) {
		//Add the contribution from time step t+W.
		crowd_association_kernel<<<(N/threadNum)+1, threadNum>>>(d_crowd_association, d_crowd_assmag_sq, d_partition, t+W, N, 1);
			
		//Compute the similarity and the distance for each vertex.
		comm_simmeasure_kernel<<<(N/threadNum)+1, threadNum>>>(d_comm_cossim, d_comm_eucdis, d_comm_association, d_comm_assmag_sq, d_crowd_association, d_crowd_assmag_sq, N, t, W);

		//Remove the contributions of time step t-W as it will no longer be needed.
		crowd_association_kernel<<<(N/threadNum)+1, threadNum>>>(d_crowd_association, d_crowd_assmag_sq, d_partition, t-W, N, -1);
	}

	

	cudaMemcpy(h_segments.comm_cossim, d_comm_cossim, sizeof(float)*N*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_segments.comm_eucdis, d_comm_eucdis, sizeof(float)*N*T, cudaMemcpyDeviceToHost);

	cudaFree(d_comm_association);
	cudaFree(d_comm_assmag_sq);

	cudaFree(d_crowd_association);
	cudaFree(d_crowd_assmag_sq);

	cudaFree(d_comm_cossim);
	cudaFree(d_comm_eucdis);

	cudaFree(d_partition);

	cudaFree(d_communities);
}

void save_features(const std::string dir, std::int32_t *h_features, const std::uint32_t N, const std::uint32_t T, const std::int32_t F) {
	std::string fpath = dir + "/features.dat";
	FILE *file = std::fopen(fpath.c_str(), "wb");
		fwrite(&N, sizeof(std::uint32_t), 1, file);
		fwrite(&T, sizeof(std::uint32_t), 1, file);
		fwrite(&F, sizeof(std::int32_t), 1, file);
		fwrite(h_features, sizeof(std::int32_t), N*T, file);
	fclose(file);
}


void save_segments(const std::string dir, std::int32_t *h_segments, const std::uint32_t N, const std::uint32_t T) {
	std::string fpath = dir + "/segments.dat";
	FILE *file = std::fopen(fpath.c_str(), "wb");
		fwrite(&N, sizeof(std::uint32_t), 1, file);
		fwrite(&T, sizeof(std::uint32_t), 1, file);
		fwrite(h_segments, sizeof(std::int32_t), N*T, file);
	fclose(file);
}

void save_memberships(const std::string dir, std::int16_t *h_memberships, const std::uint32_t N, const std::uint32_t T) {
	std::string fpath = dir + "/memberships.dat";
	FILE *file = std::fopen(fpath.c_str(), "wb");
		fwrite(&N, sizeof(std::uint32_t), 1, file);
		fwrite(&T, sizeof(std::uint32_t), 1, file);
		fwrite(h_memberships, sizeof(std::int16_t), N*T, file);
	fclose(file);
}

void save_magnetisation(const std::string dir, MAGNETISATION &mag) {
	std::string fpath =  dir + "/magnetisation.dat";

	std::uint32_t R = 1;

	if (FileExists(fpath)) {
		FILE *file = std::fopen(fpath.c_str(), "rb+");
			fread(&R, sizeof(std::uint32_t), 1, file);
			fseek(file, 0, SEEK_SET);
			R++;
			fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}
	else {
		FILE *file = std::fopen(fpath.c_str(), "wb");
			fwrite(&R, sizeof(std::uint32_t), 1, file);
			fwrite(&mag.N, sizeof(std::uint32_t), 1, file);
			fwrite(&mag.T, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}

	FILE *file = std::fopen(fpath.c_str(), "ab");
	fwrite(mag.h_gmags, sizeof(std::int16_t), mag.T, file);
	fclose(file);
}

void save_crowdSizeDists(const std::string dir, const std::uint32_t interval, jaggedlist<std::uint16_t> &h_crowdsizes) {
	size_t Ts = h_crowdsizes.size();
	std::uint32_t t;
	std::string tmpdir = dir + "/crowd_sizes";
	dirCreate(tmpdir);

	for (size_t i = 0; i < Ts; i++) {
		t = i * interval;

		std::string fpath = tmpdir + "/t_" + zeroPadNum(std::to_string(t), 6) + ".dat";
		

		std::uint32_t R = 1;
		std::uint32_t C = 0;

		if (FileExists(fpath)) {
			FILE *file = std::fopen(fpath.c_str(), "rb+");
				fseek(file, sizeof(std::uint32_t), SEEK_SET);
				fread(&R, sizeof(std::uint32_t), 1, file);
				fread(&C, sizeof(std::uint32_t), 1, file);
				fseek(file, sizeof(std::uint32_t), SEEK_SET);
				R++;
				fwrite(&R, sizeof(std::uint32_t), 1, file);
				C += (std::uint32_t) h_crowdsizes[i].size();
				fwrite(&C, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
		else {
			FILE *file = fopen(fpath.c_str(), "wb");
				fwrite(&t, sizeof(std::uint32_t), 1, file);
				fwrite(&R, sizeof(std::uint32_t), 1, file);
				C = (std::uint32_t) h_crowdsizes[i].size();
				fwrite(&C, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
		

		FILE *file = fopen(fpath.c_str(), "ab");
			C = (std::uint32_t) h_crowdsizes[i].size();
			fwrite(&h_crowdsizes[i][0], sizeof(std::uint16_t), C, file);
		fclose(file);
	}
}

void save_lifetimes(const std::string dir, const std::string name, const std::uint32_t S, const std::uint32_t ML, std::uint32_t *h_lifetimes) {
	std::string fpath = dir + "/" + name + ".dat";
	std::uint32_t R = 1;

	if (FileExists(fpath)) {
		FILE *file = std::fopen(fpath.c_str(), "rb+");
		fread(&R, sizeof(std::uint32_t), 1, file);
		fseek(file, 0, SEEK_SET);
		R++;
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}
	else {
		FILE *file = std::fopen(fpath.c_str(), "wb");
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}

	FILE *file = std::fopen(fpath.c_str(), "ab");
		fwrite(&S, sizeof(std::uint32_t), 1, file);
		fwrite(&ML, sizeof(std::uint32_t), 1, file);
		fwrite(h_lifetimes, sizeof(std::uint32_t), S, file);
	fclose(file);
}

void save_pathlengths(const std::string dir, SEGMENTS &h_segments) {
	printf("\t\tSaving Results...");
	dirCreate(dir);

	std::vector<std::string> fpath;
	fpath.push_back(dir + "/segment_apl.dat");
	fpath.push_back(dir + "/segment_apl_min.dat");
	fpath.push_back(dir + "/segment_apl_max.dat");
	fpath.push_back(dir + "/segment_vpl.dat");
	fpath.push_back(dir + "/segment_mpl.dat");
	fpath.push_back(dir + "/segment_pop.dat");
	fpath.push_back(dir + "/segment_dist.dat");
	fpath.push_back(dir + "/segment_modularities.dat");


	const size_t numOfFiles = 8;

	FILE *files[numOfFiles];

	for (size_t f = 0; f < numOfFiles; f++) {
		std::uint32_t S = 0;
		std::uint32_t R = 0;
		if (FileExists(fpath[f])) {
			FILE *file = std::fopen(fpath[f].c_str(), "rb+");
			fread(&S, sizeof(std::uint32_t), 1, file);
			fread(&R, sizeof(std::uint32_t), 1, file);
			fseek(file, 0, SEEK_SET);
			S += h_segments.S;
			R++;
			fwrite(&S, sizeof(std::uint32_t), 1, file);
			fwrite(&R, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
		else {
			FILE *file = std::fopen(fpath[f].c_str(), "wb");
			S = h_segments.S;
			R = 1;
			fwrite(&S, sizeof(std::uint32_t), 1, file);
			fwrite(&R, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
	}

	std::uint32_t S = h_segments.S;

	for (size_t f = 0; f < numOfFiles; f++) {
		files[f] = std::fopen(fpath[f].c_str(), "ab");
	}

	printf("Number of Segments = %d\n", h_segments.S);
	std::uint16_t L;
	for (size_t s = 0; s < S; s++) {
		L = h_segments.lifetimes[s];
		for (size_t i = 0; i < numOfFiles; i++) {
			fwrite(&L, sizeof(std::uint16_t), 1, files[i]);
		}

		fwrite(&(h_segments.apl[s][0]),				sizeof(float),			L, files[0]);
		fwrite(&(h_segments.apl_min[s][0]),			sizeof(float),			L, files[1]);
		fwrite(&(h_segments.apl_max[s][0]),			sizeof(float),			L, files[2]);
		fwrite(&(h_segments.vpl[s][0]),				sizeof(float),			L, files[3]);
		fwrite(&(h_segments.mpl[s][0]),				sizeof(float),			L, files[4]);
		fwrite(&(h_segments.pop[s][0]),				sizeof(std::uint16_t),	L, files[5]);
		fwrite(&(h_segments.dist_from_start[s][0]), sizeof(float),			L, files[6]);
		fwrite(&(h_segments.mods[s][0]),			sizeof(float),			L, files[7]);
	}

	for (size_t f = 0; f < numOfFiles; f++) {
		fclose(files[f]);
	}

	S = 0;
	std::uint32_t R = 0;
	std::string lfts_path = dir + "/lifetimes.dat";
	if (FileExists(lfts_path)) {
		FILE *file = std::fopen(lfts_path.c_str(), "rb+");
		fread(&S, sizeof(std::uint32_t), 1, file);
		fread(&R, sizeof(std::uint32_t), 1, file);
		fseek(file, 0, SEEK_SET);
		S += h_segments.S;
		R++;
		fwrite(&S, sizeof(std::uint32_t), 1, file);
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}
	else {
		FILE *file = std::fopen(lfts_path.c_str(), "wb");
		S = h_segments.S;
		R = 1;
		fwrite(&S, sizeof(std::uint32_t), 1, file);
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}

	S = h_segments.S;
	FILE *lfts_file = std::fopen(lfts_path.c_str(), "ab");
	fwrite(h_segments.lifetimes, sizeof(std::uint16_t), S, lfts_file);
	fclose(lfts_file);
	printf("\t\tSave Complete\n");
}

void save_pathlengths_v2(const std::string dir, SEGMENTS &h_segments) {
	printf("\t\tSaving Results...");
	dirCreate(dir);

	std::vector<std::string> fpath;
	fpath.push_back(dir + "/segment_apl.v2.dat");
	fpath.push_back(dir + "/segment_apl_min.v2.dat");
	fpath.push_back(dir + "/segment_apl_max.v2.dat");
	fpath.push_back(dir + "/segment_vpl.v2.dat");
	fpath.push_back(dir + "/segment_mpl.v2.dat");
	fpath.push_back(dir + "/segment_pop.v2.dat");
	fpath.push_back(dir + "/segment_dist.v2.dat");
	fpath.push_back(dir + "/segment_modularities.v2.dat");

	const size_t numOfFiles = 8;

	FILE *files[numOfFiles];

	for (size_t f = 0; f < numOfFiles; f++) {
		std::uint32_t R = 0;
		if (FileExists(fpath[f])) {
			FILE *file = std::fopen(fpath[f].c_str(), "rb+");
			fread(&R, sizeof(std::uint32_t), 1, file);
			fseek(file, 0, SEEK_SET);
			R++;
			fwrite(&R, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
		else {
			FILE *file = std::fopen(fpath[f].c_str(), "wb");
			R = 1;
			
			fwrite(&R, sizeof(std::uint32_t), 1, file);
			fclose(file);
		}
	}

	std::uint32_t S = h_segments.S;

	for (size_t f = 0; f < numOfFiles; f++) {
		files[f] = std::fopen(fpath[f].c_str(), "ab");
		fwrite(&S, sizeof(std::uint32_t), 1, files[f]);
	}

	printf("Number of Segments = %d\n", h_segments.S);
	std::uint16_t L;
	for (size_t s = 0; s < S; s++) {
		L = h_segments.lifetimes[s];
		for (size_t i = 0; i < numOfFiles; i++) {
			fwrite(&L, sizeof(std::uint16_t), 1, files[i]);
		}

		fwrite(&(h_segments.apl[s][0]),				sizeof(float),			L, files[0]);
		fwrite(&(h_segments.apl_min[s][0]),			sizeof(float),			L, files[1]);
		fwrite(&(h_segments.apl_max[s][0]),			sizeof(float),			L, files[2]);
		fwrite(&(h_segments.vpl[s][0]),				sizeof(float),			L, files[3]);
		fwrite(&(h_segments.mpl[s][0]),				sizeof(float),			L, files[4]);
		fwrite(&(h_segments.pop[s][0]),				sizeof(std::uint16_t),	L, files[5]);
		fwrite(&(h_segments.dist_from_start[s][0]), sizeof(float),			L, files[6]);
		fwrite(&(h_segments.mods[s][0]),			sizeof(float),			L, files[7]);
	}

	for (size_t f = 0; f < numOfFiles; f++) {
		fclose(files[f]);
	}

	S = 0;
	std::uint32_t R = 0;
	std::string lfts_path = dir + "/lifetimes.dat";
	if (FileExists(lfts_path)) {
		FILE *file = std::fopen(lfts_path.c_str(), "rb+");
		fread(&R, sizeof(std::uint32_t), 1, file);
		fseek(file, 0, SEEK_SET);
		R++;
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}
	else {
		FILE *file = std::fopen(lfts_path.c_str(), "wb");
		R = 1;
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}

	S = h_segments.S;
	FILE *lfts_file = std::fopen(lfts_path.c_str(), "ab");
	fwrite(&S, sizeof(std::uint32_t), 1, lfts_file);
	fwrite(h_segments.lifetimes, sizeof(std::uint16_t), S, lfts_file);
	fclose(lfts_file);
	printf("\t\tSave Complete\n");
}

void save_community_associations(const std::string dname, std::string fname, const size_t mixing_time, const size_t W, SEGMENTS &h_segments) {
	dirCreate(dname);

	const size_t WSz = 2 * W + 1;
	std::string fpath = dname + "/" + fname + "_W" + zeroPadNum(std::to_string(WSz), 3) + ".dat";

	std::uint32_t R = 0;
	std::uint32_t N = h_segments.N;
	std::uint32_t T = h_segments.T - mixing_time - 1;

	if (FileExists(fpath)) {
		FILE *file = std::fopen(fpath.c_str(), "rb+");
		fread(&R, sizeof(std::uint32_t), 1, file);
		fseek(file, 0, SEEK_SET);
		R++;
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fclose(file);
	}
	else {
		FILE *file = std::fopen(fpath.c_str(), "wb");
		R = 1;
		fwrite(&R, sizeof(std::uint32_t), 1, file);
		fwrite(&N, sizeof(std::uint32_t), 1, file);
		fwrite(&T, sizeof(std::uint32_t), 1, file);

		fclose(file);
	}

	FILE *file = std::fopen(fpath.c_str(), "ab");	
		fwrite(h_segments.comm_cossim + mixing_time*N, sizeof(float), N*T, file);
	fclose(file);
}
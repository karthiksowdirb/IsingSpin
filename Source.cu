#include "../../../FileIO.h"
#include <string>

#include "IsingSpin.cuh"
#include "Quench.h"
#include "Analysers.cuh"

#include <cuda.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <Windows.h>

using namespace std;

void save(const std::string dir, const std::uint32_t N, const std::uint32_t T, const std::int8_t *h_spins);

int main(void) {
	SIMPARAMETERS simpars = load_parameters();

	//Alterable Parameters
	std::string dname = simpars.dname;
	//dname += "/P_" + to_string(simpars.P);

	printf("%s\n", dname.c_str());

	const size_t Realisations = simpars.Realisations;

	size_t mix_time = simpars.mix_time;
	size_t relax_time = simpars.relax_time;
	size_t analysis_time = simpars.analysis_time;
	QUENCHTYPE qtype = (QUENCHTYPE) simpars.qtype;

	std::chrono::high_resolution_clock::time_point finish_ct;
	std::chrono::high_resolution_clock::time_point start_ct = std::chrono::high_resolution_clock::now();
	int prev30 = 0;

	for (double BETA = simpars.initBETA; BETA < simpars.finBETA; BETA += simpars.stepBETA) {
		std::printf("BETA = %f\n", BETA);
		//Resultant parameters
		QUENCH quench;
		quench.type = qtype;
		quench.init_beta = 0.;
		quench.final_beta = BETA;

		size_t quench_time = quench.time_taken();

		size_t total_time = mix_time + relax_time + quench_time;

		//Load the graph
		GRAPH graph;
		std::vector<std::string> g_paths;
		searchDirs(dname, ".mat", g_paths);

		jaggedlist<std::uint16_t> h_crowdsizes;

		for (size_t gg = 0; gg < g_paths.size(); gg++) {
			std::printf("%s\n", g_paths[gg].c_str());

			load_graph(g_paths[gg], graph);
			const std::uint32_t N = graph.NoV;

			std::int32_t *g_apsp = new std::int32_t[N*N]();
			graph_apsp(g_apsp, graph);

			//Data arrays
			std::int8_t *h_spins, *d_spins;
			h_spins = new std::int8_t[N*total_time]();
			cudaMalloc((void**)&d_spins, sizeof(std::int8_t)*N*total_time);
			cudaMemcpy(d_spins, h_spins, sizeof(std::int8_t)*N*total_time, cudaMemcpyHostToDevice);

			std::int32_t *h_features;
			h_features = new std::int32_t[N*total_time]();
			std::int16_t *h_memberships = new std::int16_t[N*total_time]();


			float *h_memberships_apl, *h_memberships_mpl;
			h_memberships_apl = new float[N*total_time]();
			h_memberships_mpl = new float[N*total_time]();

			std:string gdir = boost::filesystem::path(g_paths[gg]).parent_path().parent_path().string();
			std::string betadir = gdir + "/results/beta_" + zeroPadNum(std::to_string(quench.final_beta), 2);
			//printf("%s\n", betadir.c_str());
			dirCreate(betadir);

			//Ising Spin Simulation.
			IsingSpin ismodel;

			ismodel.set_graph(graph);

			std::chrono::high_resolution_clock::time_point finish_t;
			std::chrono::high_resolution_clock::time_point start_t = std::chrono::high_resolution_clock::now();
			for (size_t r = 0; r < Realisations; r++) {
				std::printf("///////////////////////////////////\n");
				std::printf("%d / %d || %d / %d\n", r + 1, Realisations, gg + 1, g_paths.size());

				ismodel.run(d_spins, mix_time, quench_time, relax_time, quench);
				//printf("\tRun Complete\n");

				MAGNETISATION mag;
				mag.N = graph.NoV;
				mag.T = total_time;
				mag.h_gmags = new std::int16_t[total_time]();
				magnetisation(mag, d_spins);
				save_magnetisation(betadir, mag);
				delete[] mag.h_gmags;

				//printf("\tMagnetisation Complete\n");

				cudaMemcpy(h_spins, d_spins, sizeof(std::int8_t)*N*total_time, cudaMemcpyDeviceToHost);

				noise_remover(d_spins, N, total_time, 10);

				cudaMemcpy(h_spins, d_spins, sizeof(std::int8_t)*N*total_time, cudaMemcpyDeviceToHost);
				
				/*
				dirCreate(betadir + "/temp");
				save(betadir, N, total_time, h_spins);
				*/

				std::printf("\tNoise Removed\n");


				SEGMENTS h_segments;
				h_segments.communities = load_community(fileDirectory(g_paths[gg]));
				h_segments.init_partitions(N, total_time);

				std::uint32_t S = identify_segments2(h_segments, h_memberships, h_memberships_apl, h_memberships_mpl, d_spins, h_spins, graph, N, total_time, mix_time + analysis_time);

				h_segments.init_lifetimes(S);


				compute_lifetimes(h_segments, mix_time + analysis_time);

				h_segments.init_pathlengths(relax_time - analysis_time);

				compute_pathlengths(h_segments, h_memberships_apl, h_memberships_mpl, g_apsp, mix_time + analysis_time);

				compute_modularities(h_segments, graph, mix_time + analysis_time);

				save_pathlengths_v2(betadir + "/" + std::to_string(relax_time), h_segments);
				//printf("\tSegment Lifetimes Measured\n");

				if (h_segments.communities != nullptr && simpars.W >= 0) {
					const size_t W = simpars.W;
					compute_association(h_segments, mix_time + analysis_time, W);

					printf("Saving Community Associations");
					std::string commassdir = betadir + "/" + std::to_string(relax_time) + "/associations/";
					save_community_associations(commassdir, "graph_" + zeroPadNum(std::to_string(gg), 3), mix_time + analysis_time, W, h_segments);
				}

				finish_t = std::chrono::high_resolution_clock::now();
				auto delta_t = std::chrono::duration_cast<std::chrono::seconds>(finish_t - start_t).count();
				std::printf("\t\tBETA = %f\t\tTime Taken = %d\n", BETA, delta_t);
				std::printf("Number of Segments = %d\n", S);
			}


			delete[] h_features;
			delete[] h_memberships;
			delete[] h_memberships_apl;
			delete[] h_memberships_mpl;

			delete[] h_spins;
			delete[] g_apsp;

			cudaFree(d_spins);

			auto delta_t = std::chrono::duration_cast<std::chrono::minutes>(finish_t - start_t).count();

			finish_ct = std::chrono::high_resolution_clock::now();
			auto delta_ct = std::chrono::duration_cast<std::chrono::minutes>(finish_ct - start_ct).count();
			std::printf("Total Time Taken (minutes): %d | %d\n\n", delta_t, delta_ct);

			/*
			int curr30 = delta_ct / 45;
			if (curr30 != prev30) {
				Sleep(1000 * 60 * 5);
				prev30 = curr30;
			}
			*/

			cudaDeviceSynchronize();
			cudaDeviceReset();
		}
	}
	std::printf("\n");
	
	return 0;
}

void save(const std::string dir, const std::uint32_t N, const std::uint32_t T, const std::int8_t *h_spins) {

	FILE *file = fopen((dir + "/ising_spin.dat").c_str(), "wb");

	fwrite(&N, sizeof(std::uint32_t), 1, file);
	fwrite(&T, sizeof(std::uint32_t), 1, file);
	fwrite(h_spins, sizeof(std::int8_t), N*T, file);

	fclose(file);
}
#ifndef ANALYSERS_CUH
#define ANALYSERS_CUH

#include <cstdint>
#include <vector>
#include <algorithm>
#include "../../../FileIO.h"

template <class T>
class jaggedlist {
	std::vector< std::vector<T> > data;
public:

	std::vector<T>& operator[](const size_t i) {
		return data[i]; 
	}

	void clear() {
		data.clear();
	}

	size_t width(const size_t i) const {
		return data[i].size();
	}

	size_t max_width() {
		size_t mx = 0;
		for (size_t i = 0; i < data.size(); i++) {
			mx = (mx < data[i].size()) ? (data[i].size()) : (mx);
		}
		return mx;
	}

	void set_size(const size_t s) {
		data.resize(s);
	}

	size_t size() {
		return data.size();
	}
};

struct MEMBERSHIPS {
	size_t N;
	size_t C;
	size_t T;

	std::uint16_t *memberships;
};

struct MAGNETISATION {
	std::uint32_t N;
	std::uint32_t T;
	std::uint32_t C;
	std::int16_t* h_gmags;
	std::int16_t* h_cmags;
};

struct SEGMENTS {
	std::uint32_t N;
	std::uint32_t T;

	std::uint32_t ML;
	std::uint32_t S;

	std::int32_t *partitions;
	

	std::int32_t *birthdeath;
	std::uint16_t *lifetimes;

	float **apl;				//Average of the segment's pathlengths.
	float **apl_min;			//The smallest average pathlength.
	float **apl_max;			//The largest average pathlength.
	float **vpl;				//Variance of the segment's pathlengths 
	float **mpl;				//Largest pathlength in the segment (Diameter).
	float **cc_means;			//The mean internal clustering coefficients of each segment.
	float **cc_vars;			//The variance in the internal clustering coefficient of each segment.

	float **mods;				//modularities.
	std::uint16_t **intdegs;	//Number of internal degrees
	std::uint16_t **extdegs;	//Number of external degrees.

	std::uint16_t **pop; //Number of vertices in the segment at each time t.

	std::int32_t *centre; //Vertex which is the centre of the segment at each time t.
	float **dist_from_start; //Shortest pathlength from the first centre to the current centre.


	bool community_loaded;
	std::uint16_t *communities;

	float *comm_cossim;
	float *comm_eucdis;

	SEGMENTS() {
		N = T = S = ML = 0;

		partitions	= nullptr;
		birthdeath	= nullptr;
		lifetimes	= nullptr;

		apl = nullptr;
		apl_min = nullptr;
		apl_max = nullptr;
		vpl = nullptr;
		mpl = nullptr;
		cc_means = nullptr;
		cc_vars = nullptr;

		mods = nullptr;
		intdegs = nullptr;
		extdegs = nullptr;

		pop = nullptr;

		centre = nullptr;
		dist_from_start = nullptr;

		community_loaded = false;
		communities = nullptr;

		comm_cossim = nullptr;
		comm_eucdis = nullptr;
	}

	~SEGMENTS() {
		if (partitions != nullptr) {
			delete[] partitions;
		}
		
		if (birthdeath != nullptr) {
			delete[] birthdeath;
		}

		if (lifetimes != nullptr) {
			delete[] lifetimes;
		}
		if (centre != nullptr) {
			delete[] centre;
		}

		for (size_t s = 0; s < S; s++) {
			delete[] apl[s];
			delete[] apl_min[s];
			delete[] apl_max[s];
			delete[] vpl[s];
			delete[] mpl[s];
			delete[] dist_from_start[s];
			delete[] cc_means[s];
			delete[] cc_vars[s];
			delete[] intdegs[s];
			delete[] extdegs[s];
			delete[] mods[s];
		}

		

		delete[] apl;
		delete[] apl_min;
		delete[] apl_max;
		delete[] vpl;
		delete[] mpl;
		delete[] cc_means;
		delete[] cc_vars;
		delete[] mods;
		delete[] intdegs;
		delete[] extdegs;

		delete[] pop;
		delete[] dist_from_start;

		delete[] communities;

		delete[] comm_cossim;
		delete[] comm_eucdis;
	}

	void init_partitions(const uint32_t N, const uint32_t T) {
		this->N = N;
		this->T = T;
		partitions = new std::int32_t[N*T]();

		comm_cossim = new float[N*T]();
		comm_eucdis = new float[N*T]();
	}

	void init_lifetimes(const uint32_t S) {
		this->S = S;
		lifetimes = new std::uint16_t[S]();
		birthdeath = new std::int32_t[2 * S]();
	}

	void init_pathlengths(const uint32_t ML) {
		this->ML = ML;
		apl = new float*[S];
		apl_min = new float*[S];
		apl_max = new float*[S];
		vpl = new float*[S];
		mpl = new float*[S];
		cc_means = new float*[S];
		cc_vars = new float*[S];

		mods = new float*[S];
		intdegs = new std::uint16_t*[S];
		extdegs = new std::uint16_t*[S];

		pop = new std::uint16_t*[S];
		centre = new std::int32_t[S];
		std::fill(centre, centre + S, -1);
		dist_from_start = new float*[S];

		std::int32_t lfts;
		for (size_t s = 0; s < S; s++) {
			lfts = lifetimes[s];
			apl[s] = new float[lfts]();
			vpl[s] = new float[lfts]();
			mpl[s] = new float[lfts]();

			apl_min[s] = new float[lfts];
			apl_max[s] = new float[lfts]();
			std::fill(apl_min[s], apl_min[s] + lfts, (float)(N + 1));

			cc_means[s] = new float[lfts]();
			cc_vars[s] = new float[lfts]();

			mods[s] = new float[lfts]();
			intdegs[s] = new std::uint16_t[lfts]();
			extdegs[s] = new std::uint16_t[lfts]();

			pop[s] = new std::uint16_t[lfts]();

			dist_from_start[s] = new float[lfts]();
		}
	}
};

void graph_apsp(std::int32_t *g_apsp, GRAPH &h_graph);

void noise_remover(std::int8_t *d_spin, const size_t N, const size_t T, const size_t window);

void magnetisation(MAGNETISATION &mag, std::int8_t *d_spin);

void crowdSizeDists(jaggedlist<std::uint16_t> &h_crowdsizes, std::int8_t *h_spin, GRAPH &h_graph, const size_t N, const size_t T, const size_t interval);

std::int32_t identify_features(std::int32_t *h_features, std::int8_t *h_spin, GRAPH &h_graph, const size_t N, const size_t T);

std::uint32_t identify_segments(SEGMENTS &h_segments, std::int16_t *h_memberships, float* h_memberships_apl, float* h_memberships_mpl, std::int8_t *h_spins, GRAPH &h_graph, const std::uint32_t N, const std::uint32_t T, const std::uint32_t mixing_time);

std::uint32_t identify_segments2(SEGMENTS &h_segments, std::int16_t *h_memberships, float* h_memberships_apl, float* h_memberships_mpl, std::int8_t *d_spins, std::int8_t *h_spins, GRAPH &h_graph, const std::uint32_t N, const std::uint32_t T, const std::uint32_t mixing_time);

void compute_lifetimes(SEGMENTS &h_segments, const std::uint32_t mixing_time);

void compute_pathlengths(SEGMENTS &h_segments, float *h_memberships_apl, float *h_memberships_mpl, std::int32_t *g_apsp, const size_t analysis_time);

void compute_segment_props(SEGMENTS &h_segments, GRAPH &h_graph);

void compute_community_properties(COMMUNITY_DATA &comm, SEGMENTS &h_segments, std::int8_t *d_spins, const size_t N, const size_t T, const size_t mix_time);

void compute_membership_device(std::int16_t *d_memberships, std::uint16_t *d_memberships_count, std::int8_t *d_spins, std::uint16_t *d_deglist, std::uint16_t *d_adjlist, const size_t N, const size_t K, const size_t T, const size_t analysis_time);

void compute_modularities(SEGMENTS &h_segments, GRAPH &h_graph, const size_t analysis_time);

void compute_association(SEGMENTS &h_segments, const size_t analysis_time, const size_t W);

//Save Functions

void save_features(const std::string dir, std::int32_t *h_features, const std::uint32_t N, const std::uint32_t T, const std::int32_t F);

void save_segments(const std::string dir, std::int32_t *h_segments, const std::uint32_t N, const std::uint32_t T);

void save_memberships(const std::string dir, std::int16_t *h_memberships, const std::uint32_t N, const std::uint32_t T);

void save_magnetisation(const std::string dir, MAGNETISATION &mag);

void save_crowdSizeDists(const std::string dir, const std::uint32_t interval, jaggedlist<std::uint16_t> &h_crowdsizes);

void save_lifetimes(const std::string dir, const std::string name, const std::uint32_t S, const std::uint32_t ML, std::uint32_t *h_lifetimes);

void save_pathlengths(const std::string dir, SEGMENTS &h_segments);

void save_pathlengths_v2(const std::string dir, SEGMENTS &h_segments);

void save_community_associations(const std::string dname, std::string fname, const size_t mixing_time, const size_t W, SEGMENTS &h_segments);

#endif
/**
 *  GPU Monte Carlo Scintillator Simulation
 *
 *  (C) 2009 Attila Incze <attila.incze@gmail.com>
 *  http://atimb.me
 *
 *  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this license, visit
 *  http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.
 * 
 */

#ifndef _SCINTILLATOR_KERNEL_H_
#define _SCINTILLATOR_KERNEL_H_

#include <stdio.h>
#include "cutil_math.h"
#include "device_functions.h"

#define PI                 3.14159265358979f
#define SQRT(x)	           __powf(x, 0.5f)
#define SIN(x)	           __sinf(x)
#define COS(x)	           __cosf(x)
#define LOG(x)	           __logf(x)
#define LOG10(x)           __log10f(x)
#define REST_E	           0.511f

// LCG generator parameters
#define G                  3249286849523012805l
#define C                  1l
#define M                  9223372036854775808l

// Cross section data
texture<float4, 1, cudaReadModeElementType> crossSection;


struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};


// LCG random generator
__device__
float getRandom(unsigned long long int &seed) {
	seed = (seed * G + C ) & (M-1);
	return (float)seed / M;
}


// Generates a random, isotrop direction
__device__
float3 isotropDirection(unsigned long long int &seed) {
	float3 ret;
	float alpha = 2.0f * PI * getRandom(seed);
	ret.x = 2.0f * getRandom(seed) - 1.0f;
	float rho = SQRT( 1.0f - ret.x * ret.x );
	ret.y = rho * SIN( alpha );
	ret.z = rho * COS( alpha );
	return ret;
}


// Intersects a ray with a box
__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six box planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


// Simulate a Compton-effect (Kahn method)
// Returns the detected energy
__device__
float doCompton(Ray &gammaRay, float &energy, unsigned long long int &seed) {
	float eRatio;
	float inverseLambda = energy / REST_E;
	while (true) {
		float r1 = getRandom(seed);
		float r2 = getRandom(seed);
		float r3 = getRandom(seed);
		if (r1 <= (1.0f + 2.0f * inverseLambda) / (9.0f + 2.0f * inverseLambda)) {
			eRatio = 1.0f + 2.0f * r2 * inverseLambda;
			if (r3 <= 4.0f * (eRatio - 1.0f) / (eRatio * eRatio)) {
				break;
			}
		} else {
			eRatio = (1.0f + 2.0f * inverseLambda) / (1.0f + 2.0f * r2 * inverseLambda);
			float d = 1.0f / inverseLambda;
			d *= 1.0f - eRatio;
			d += 1.0f;
			if (r3 <= 0.5f * (d * d + 1.0f / eRatio)) {
				break;
			}
		}
	}
	float r1 = 2.0f * PI * getRandom(seed);
	float z = 1.0f + (1.0f - eRatio) / inverseLambda;
	float sinAlpha = SQRT(1 - z * z);
	float x = sinAlpha * COS(r1);
	float y = sinAlpha * SIN(r1);
	float u = gammaRay.d.x;
	float v = gammaRay.d.y;
	float w = gammaRay.d.z;
	float rho = SQRT(u * u + v * v);
	float inverseRho = 1 / rho;
	float d = y * w;
	gammaRay.d.x = (d * u + x * v) * inverseRho + u * z;
	gammaRay.d.y = (d * v - x * u) * inverseRho + v * z;
	gammaRay.d.z = rho * y + w * z;
	d = energy / eRatio;
	float d2 = energy - d;
	energy = d;
	return (d2);
}


// Calculates the reaction that happens
__device__
int csGetReaction(float4 cs, unsigned long long int &seed) {
	float r = getRandom(seed) * cs.w;
	if (cs.x >= r)
		return 1;
	else if ((cs.x+cs.y) >= r)
		return 2;
	else
		return 3;
}


// Get Mean Free Path
__device__
float getMFP(float rho, float total, unsigned long long int &seed) {
	return (-LOG(1.0f - getRandom(seed)) / (total * rho));
}


// Apply scatter to absorbed energy, and save it to device memory
__device__
void detectE(float detectedE, const float scatter, int* const result, const int tid, float maxE, unsigned long long int &seed) {
	// gauss scatter
	if (scatter > 0.0f) {
		float gauss = -6.0f;
		for (int i=0; i<12; ++i)
			gauss += getRandom(seed);
		gauss *= scatter;
		detectedE += gauss;
	}
	// calculate which channel it is
	int index = (int)( ((detectedE)/(maxE) * 1024.999f)-0.5f);
	// save value to memory
	result[tid] = index;
}


// Sets photon launch parameters
__device__
void newPhoton(int &all, Ray &gammaRay, float &detectedE, const float3 origin, float &energy, const float oenergy, bool &pair_exists, unsigned long long int &seed) {
	// Create new photon
	pair_exists = false;
	gammaRay.o = origin;
	gammaRay.d = isotropDirection(seed);
	detectedE = 0.0f;
	energy = oenergy;
	++all;
}


// Sets pair-photon parameters
__device__
bool handlePair(bool &pair_exists, Ray &gammaRay, const Ray gammaPair, float &energy) {
	if (!pair_exists)
		return false;
	energy = REST_E;
	gammaRay = gammaPair;
	pair_exists = false;
	return true;
}


// THE kernel that simulates PHOTONS_IN_THREAD photons in a row then exits
__global__
void launchPhotons(const float3 origin, const float oenergy, const float3 boxMin, const float3 boxMax, const float rho, const float scatter, int* const result, const int PHOTONS_IN_THREAD, const long long int offset) {

	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long long int seed = tid + offset;
	Ray gammaRay, gammaPair;
	float detectedE = 0.0f;
	bool pair_exists;
	float tnear, tfar;
	int hit;
	float energy;
	int all = 0;
	bool new_photon_need = true;


	// Photon follower iteration
	//---------------------------------
	while (true) {

		// Set new photon parameters
		if (new_photon_need) {
			if (detectedE > 0.0f)
				detectE(detectedE, scatter, result, tid*PHOTONS_IN_THREAD + all-1, oenergy + 0.1f, seed);
			if (all == PHOTONS_IN_THREAD)
				return;
			newPhoton(all, gammaRay, detectedE, origin, energy, oenergy, pair_exists, seed);
		}

		// Get gamma ray intersection points with crystal
		hit = intersectBox(gammaRay, boxMin, boxMax, &tnear, &tfar);

		// Photon not in crystal
		//---------------------------------
		if (new_photon_need) {
			if (!hit || (tfar <= 0.0f )) {
				continue;
			}
			// Move photon to collision point (if photon was outside)
			if (tnear > 0.0f) {
				tfar -= tnear;
				gammaRay.o += gammaRay.d * tnear;
			}
			new_photon_need = false;
		}

		// Get cross section at photon energy
		float norm = (LOG10(energy)+3.0f)/8.0f;
		float4 cs = tex1D(crossSection, norm);
		// Get mean free path
		float mfp = getMFP(rho, cs.w, seed);

		// Reaction happens
		//---------------------------------
		if (tfar > mfp) {

			// move photon ahead
			gammaRay.o += gammaRay.d * mfp;
			// find out what happens
			int type = csGetReaction(cs, seed);

			// Photo effect
			//---------------------------------
			if (type == 1) {
				detectedE += energy;

				if (!handlePair(pair_exists, gammaRay, gammaPair, energy)) {
					new_photon_need = true;
				}

			// Compton
			//---------------------------------
			} else if (type == 2) {
				detectedE += doCompton(gammaRay, energy, seed);

			// Pair
			//---------------------------------
			} else if (type == 3) {
				detectedE += energy - 2 * REST_E;
				gammaRay.d = isotropDirection(seed);
				pair_exists = true;
				gammaPair.o = gammaRay.o;
				gammaPair.d = -gammaRay.d;
				energy = REST_E;
			}

		// Photon leaves crystal
		//---------------------------------
		} else {
			if (!handlePair(pair_exists, gammaRay, gammaPair, energy)) {
				new_photon_need = true;
			}
		}

	}

}


#endif // #ifndef _SCINTILLATOR_KERNEL_H_

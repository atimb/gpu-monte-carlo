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

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include "cutil_math.h"


#define PI                 3.14159265358979f
#define SQRT(x)			   pow(x, 0.5f)
#define SIN(x)			   sin(x)
#define COS(x)			   cos(x)
#define LOG(x)			   log(x)
#define LOG10(x)		   log10(x)
#define REST_E			   0.511f

#ifdef WIN32
	#define G                  3249286849523012805
	#define C                  1
	#define M                  9223372036854775808
#else
	#define G                  3249286849523012805LLU
	#define C                  1
	#define M                  9223372036854775808LLU
#endif


struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};


float getRandom(unsigned long long int &seed) {
	seed = (seed * G + C ) & (M-1);
	return (float)seed / M;
}


float3 isotropDirection(unsigned long long int &seed) {

	float3 ret;
	float alpha = 2.0f * PI * getRandom(seed);
	ret.x = 2.0f * getRandom(seed) - 1.0f;
	float rho = SQRT( 1.0f - ret.x * ret.x );
	ret.y = rho * SIN( alpha );
	ret.z = rho * COS( alpha );
	return ret;

}


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


// Simulate a Compton-effect
// Returns the decected energy
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
	float inverseRho = 1.0f / rho;
	float d = y * w;
	gammaRay.d.x = (d * u + x * v) * inverseRho + u * z;
	gammaRay.d.y = (d * v - x * u) * inverseRho + v * z;
	gammaRay.d.z = rho * y + w * z;
	d = energy / eRatio;
	float d2 = energy - d;
	energy = d;
	return (d2);

}


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
float getMFP(float rho, float total, unsigned long long int &seed) {
	return (-LOG(1.0f - getRandom(seed)) / (total * rho));
}


float4 getCS(float4* csArray, float x) {
	x *= 127.0f;
	if (x <= 0.0f)
		return csArray[0];
	if (x >= 127.0f)
		return csArray[127];

	short fl = (short)floor(x);
	float rem = x - fl;
	return (csArray[fl] * (1-rem) + csArray[fl+1] * rem);
}


int launchPhoton(float3 origin, float energy, float3 boxMin, float3 boxMax, float rho, float scatter, unsigned long long int seed, float4* csArray) {

	float detectedE = 0;
	float e1 = 0.0f, e2 = energy + 0.1f;

	// Create new photon
	Ray gammaRay, gammaPair;
	bool pair_exists = false;
	gammaRay.o = origin;
	gammaRay.d = isotropDirection(seed);

	// Find intersection with scintillator crystal
	float tnear, tfar;
	int hit = intersectBox(gammaRay, boxMin, boxMax, &tnear, &tfar);

	if (hit && (tnear > 0.0f)) {
		tfar -= tnear;
		gammaRay.o += gammaRay.d * tnear;
	}

	if (hit && (tfar > 0.0f))
	while (true) {

		float norm = (LOG10(energy)+3.0f)/8.0f;
		float4 cs = getCS(csArray, norm);
		float mfp = getMFP(rho, cs.w, seed);

		// Reaction happens
		if (tfar > mfp) {
			gammaRay.o += gammaRay.d * mfp;

			// find out what happens
			int type = csGetReaction(cs, seed);
			// Photo effect
			if (type == 1) {
				detectedE += energy;
				if (pair_exists) {
					energy = REST_E;
					gammaRay = gammaPair;
					pair_exists = false;
				} else
					break;
			// Compton
			} else if (type == 2) {
				detectedE += doCompton(gammaRay, energy, seed);
			// Pair
			} else if (type == 3) {
				detectedE += energy - 2 * REST_E;
				gammaRay.d = isotropDirection(seed);
				pair_exists = true;
				gammaPair.o = gammaRay.o;
				gammaPair.d = -gammaRay.d;
				energy = REST_E;
			}

			hit = intersectBox(gammaRay, boxMin, boxMax, &tnear, &tfar);
		} else {
			if (pair_exists) {
				energy = REST_E;
				gammaRay = gammaPair;
				pair_exists = false;
				hit = intersectBox(gammaRay, boxMin, boxMax, &tnear, &tfar);
			} else
				break;
		}

	}

	if (detectedE == 0)
		return 0;

	if (scatter > 0.0f) {
		// gauss scatter
		float gauss = -6.0f;
		for (int i=0; i<12; ++i)
			gauss += getRandom(seed);
		gauss *= scatter;
		detectedE += gauss;
	}

	// select channel
	int index = (int)( ((detectedE-e1)/(e2-e1) * 1024.999f)-0.5f);
	return index;

}

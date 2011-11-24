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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <scintillator_kernel.cu>

#define ASCII_LENGTH       78
#define ASCII_HEIGHT       15
#define ASCII_ART
#define CPU_MUL            10

extern
int launchPhoton(float3 origin, float energy, float3 boxMin, float3 boxMax, float rho, float scatter, unsigned long long int seed, float4* csArray);

int THREADS = 192;
int GRIDS = 1024;
int REPEAT = 1;
int PHOTONS_IN_THREAD = 100;

float3 origin = make_float3(0.0f, 0.0f, -2.0f);
float3 boxmin = make_float3(-1.0f, -1.0f, -1.0f);
float3 boxmax = make_float3(1.0f, 1.0f, 1.0f);
float energy = 1.0f;
float rho = 3.67f;
float scatter = 0.0f;

dim3 grid, threads;
int *h_odata, *d_odata;
size_t mem_size;
unsigned int timer = 0;
float gpu_time, photonsperms, cpu_time;
int energies[1025], energies2[1025], ascii_energies[ASCII_LENGTH];
float4 *h_csData;
cudaArray *d_crossSection;


////////////////////////////////////////////////////////////////////////////////
// Handles command line parameters
////////////////////////////////////////////////////////////////////////////////
void handleParams(int argc, char** argv) {
	int parami;
	float paramf;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "printheader") ) {
		cerr << "Threads Grids PhotonsPerThread Repeat GPUTime PhotonsPerMs GPUCPURatio Energy Density SourceX SourceY SourceZ BoxminX BoxminY BoxminZ BoxmaxX BoxmaxY BoxmaxZ" << endl;
		exit(0);
	}

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "help") ) {
		cout << "\nUsage: scintillator [parameters]\n\n"<
		cout << "Set configuration with input-data.txt file in the working directory.\n\n";
		cout << "You can also use the following command line parameters:\n";
		cout << "	--threads=X    Launch the kernel with X threads\n";
		cout << "	--grids=X      Launch the kernel with X grids\n";
		cout << "	--photonsperthread=X    Simulate X photon in each thread\n";
		cout << "	--repeat=X     How many kernels to launch\n";
		cout << "	--energy=X     Photon source energy (MeV)\n";
		cout << "	--density=X    Density of the scintillator crystal (g/cm^3)\n";
		cout << "	--scatter=X    Gauss scatter FWHM to apply to measured data (MeV)\n";
		cout << "	--sourcex=X, --sourcey=Y, --sourcez=Z\n";
		cout << "	               The position of the photon source (X, Y, Z)\n";
		cout << "	--boxminx=X, --boxminy=Y, --boxminz=Z\n";
		cout << "	--boxmaxx=X, --boxmaxy=Y, --boxmaxz=Z\n";
		cout << "	               The position of the crystal (bottom left and top right corner)\n";
		cout << "	--help         Print this help\n";
		cout << "	--printheader  Prints to error output a header for results (for test running)\n";
		cout << "	--printresult  Prints to error output the result of simulation (for test running)\n";
		cout << "\nExample: ./scintillator --repeat=10 --sourcez=-5.0 --scatter=0.01\n";
		exit(0);
	}

	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "threads", &parami)) { THREADS = parami; }
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "photonsperthread", &parami)) { PHOTONS_IN_THREAD = parami; }
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "grids", &parami)) { GRIDS = parami; }
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "repeat", &parami)) { REPEAT = parami; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "energy", &paramf)) { energy = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "density", &paramf)) { rho = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "scatter", &paramf)) { scatter = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "sourcex", &paramf)) { origin.x = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "sourcey", &paramf)) { origin.y = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "sourcez", &paramf)) { origin.z = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxminx", &paramf)) { boxmin.x = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxminy", &paramf)) { boxmin.y = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxminz", &paramf)) { boxmin.z = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxmaxx", &paramf)) { boxmax.x = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxmaxy", &paramf)) { boxmax.y = paramf; }
	if (cutGetCmdLineArgumentf(argc, (const char**)argv, "boxmaxz", &paramf)) { boxmax.z = paramf; }

	printf("Simulation parameters:\n");
	printf("Source: (%.2f %.2f %.2f), %.2f MeV \n", origin.x, origin.y, origin.z, energy);
	printf("Crystal: (%.2f %.2f %.2f) - (%.2f %.2f %.2f), %.2f g/cm^3 \n", boxmin.x, boxmin.y, boxmin.z, boxmax.x, boxmax.y, boxmax.z, rho);
	printf("Threads: %d, Grids: %d, Photons per thread: %d, Repeat: %d \n", THREADS, GRIDS, PHOTONS_IN_THREAD, REPEAT);
	printf("===================================================================\n");

}

////////////////////////////////////////////////////////////////////////////////
// Loads the config from file
////////////////////////////////////////////////////////////////////////////////
void loadConfig(int argc, char** argv) {

	// Load config file
	ifstream configfile("input-data.txt");

	if (configfile.is_open()) {
		while (!configfile.eof()) {
			float f1, f2, f3;
			int i1;
			string sor;
			getline(configfile, sor);
			if (sor.find("[source]") != string::npos) {
				if (configfile >> f1 >> f2 >> f3)
					origin = make_float3(f1, f2, f3);
			} else
			if (sor.find("[boxmin]") != string::npos) {
				if (configfile >> f1 >> f2 >> f3)
					boxmin = make_float3(f1, f2, f3);
			} else
			if (sor.find("[boxmax]") != string::npos) {
				if (configfile >> f1 >> f2 >> f3)
					boxmax = make_float3(f1, f2, f3);
			} else
			if (sor.find("[energy]") != string::npos) {
				if (configfile >> f1)
					energy = f1;
			} else
			if (sor.find("[density]") != string::npos) {
				if (configfile >> f1)
					rho = f1;
			} else
			if (sor.find("[scatter]") != string::npos) {
				if (configfile >> f1)
					scatter = f1;
			} else
			if (sor.find("[threads]") != string::npos) {
				if (configfile >> i1)
					THREADS = i1;
			} else
			if (sor.find("[grids]") != string::npos) {
				if (configfile >> i1)
					GRIDS = i1;
			} else
			if (sor.find("[photonsperthread]") != string::npos) {
				if (configfile >> i1)
					PHOTONS_IN_THREAD = i1;
			} else
			if (sor.find("[repeat]") != string::npos) {
				if (configfile >> i1)
					REPEAT = i1;
			}
		}
	}

}


////////////////////////////////////////////////////////////////////////////////
// Set up variables, load cross section data, allocate memory
////////////////////////////////////////////////////////////////////////////////
void initCalculation() {
	mem_size = sizeof( int) * GRIDS * THREADS * PHOTONS_IN_THREAD;

	memset(energies, 0, sizeof(int)*1024);
	memset(energies2, 0, sizeof(int)*1024);
	memset(ascii_energies, 0, sizeof(int)*ASCII_LENGTH);

	// Cross section data
	ifstream infile("cross-section.txt");

	h_csData = (float4*) malloc( sizeof(float4)*128 );

	if (infile.is_open()) {
		for (int i=0; i<128; ++i) {
			infile >> h_csData[i].x;
			infile >> h_csData[i].y;
			infile >> h_csData[i].z;
			infile >> h_csData[i].w;
		}
	}

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cutilSafeCall(cudaMallocArray( &d_crossSection, &channelDesc, 128, 1)); 
    cutilSafeCall(cudaMemcpyToArray( d_crossSection, 0, 0, h_csData, sizeof(float4)*128, cudaMemcpyHostToDevice));

    crossSection.filterMode = cudaFilterModeLinear;   // linear interpolation
    crossSection.normalized = true;    // access with normalized texture coordinates
    crossSection.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates

    // Bind the array to the texture
    cutilSafeCall( cudaBindTextureToArray( crossSection, d_crossSection, channelDesc));

    // allocate device memory for result
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));
	cutilSafeCall( cudaMemset( d_odata, 0, mem_size));

    // setup execution parameters
    grid = dim3( GRIDS, 1, 1);
    threads = dim3( THREADS, 1, 1);

    // allocate mem for the result on host side
    h_odata = (int*) malloc( mem_size );
}


////////////////////////////////////////////////////////////////////////////////
// MC simulation on GPU
////////////////////////////////////////////////////////////////////////////////
void simulationGPU() {
	printf("Simulation on GPU in progress...\n");

	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));

	launchPhotons<<< grid, threads >>>(origin, energy, boxmin, boxmax, rho, scatter, d_odata, PHOTONS_IN_THREAD, 0);

	for (int rep=0; rep < REPEAT; ++rep) {

		cutilSafeCall( cudaThreadSynchronize() );

		// copy result from device to host
		cutilSafeCall( cudaMemcpy( h_odata, d_odata, mem_size,
									cudaMemcpyDeviceToHost) );
		// clear device memory for next run
		cutilSafeCall( cudaMemset( d_odata, 0, mem_size));

		if ((rep+1) < REPEAT)
			launchPhotons<<< grid, threads >>>(origin, energy, boxmin, boxmax, rho, scatter, d_odata, PHOTONS_IN_THREAD, (long long int)rep*THREADS*GRIDS);

		for (long long int i=0; i<GRIDS*THREADS*PHOTONS_IN_THREAD; ++i) {
			int index = h_odata[i];
			if ((index < 1025) && (index >= 0)) {
				energies[index]++;
#ifdef ASCII_ART
				int indexx = (int)(index*(ASCII_LENGTH/1025.0f));
				ascii_energies[indexx]++;
#endif
			}


		}

	}

	cutilCheckError( cutStopTimer( timer));
	gpu_time = cutGetTimerValue( timer);
	printf( "  -> GPU processing time: %.2f ms\n", gpu_time);
	photonsperms = (long long int)GRIDS*THREADS*REPEAT*PHOTONS_IN_THREAD / gpu_time;
	printf( "  -> Photons/msec: %.0f \n", photonsperms);
	cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError( cutCreateTimer( &timer));
}


////////////////////////////////////////////////////////////////////////////////
// MC simulation on CPU
////////////////////////////////////////////////////////////////////////////////
void simulationCPU() {
	printf("Simulation on CPU in progress...\n");
	cutilCheckError( cutStartTimer( timer));

	int _THREADS = THREADS*PHOTONS_IN_THREAD/CPU_MUL;
	for (int repeat = 0; repeat < REPEAT; ++repeat)
		for (int grid = 0; grid < GRIDS; ++grid)
			for (int thread = 0; thread < _THREADS; ++thread) {
				int index = launchPhoton(origin, energy, boxmin, boxmax, rho, scatter, thread+grid*_THREADS+repeat*GRIDS*_THREADS, h_csData);
				if ((index < 1025) && (index >= 0))
					energies2[index]++;
			}

	cutilCheckError( cutStopTimer( timer));
	cpu_time = cutGetTimerValue( timer)*THREADS*PHOTONS_IN_THREAD / _THREADS;
	printf( "  -> CPU processing time (estimated): %.2f ms (real: %.2f ms)\n", cpu_time, cutGetTimerValue( timer));
	printf( "  -> Photons/msec: %.0f \n", (long long int)REPEAT*GRIDS*_THREADS / cutGetTimerValue( timer));
	printf("------------------------------------\n");
	printf( "GPU/CPU ratio: %.2f \n", cpu_time / gpu_time);
	cutilCheckError( cutDeleteTimer( timer));
}

////////////////////////////////////////////////////////////////////////////////
// Store results in file, print to console
////////////////////////////////////////////////////////////////////////////////
void printResults(int argc, char** argv) {

	ofstream outfile("spectrum_GPU.txt");
	for (int i=0; i<1025; ++i)
		outfile << (float)energies[i]/(REPEAT*GRIDS*THREADS*PHOTONS_IN_THREAD) << "\n";
	outfile.close();

	ofstream outfile2("spectrum_CPU.txt");
	for (int i=0; i<1025; ++i)
		outfile2 << (float)energies2[i]/(REPEAT*GRIDS*THREADS*PHOTONS_IN_THREAD/CPU_MUL) << "\n";
	outfile2.close();

#ifdef ASCII_ART
	// Draw the fancy ascii art
	int max = 1;
	for (int i=1; i<ASCII_LENGTH; ++i)
		if (max < ascii_energies[i])
			max = ascii_energies[i];

	for (int z=ASCII_HEIGHT; z>=0; --z)
		for (int i=1; i<ASCII_LENGTH; ++i) {
			if (z == 0)
				printf("-");
			else
				if (ascii_energies[i] >= ((float)z/ASCII_HEIGHT)*max)
					printf("|"); else printf(" ");
			if (i == ASCII_LENGTH-1)
				printf("\n");
		}
#endif

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "printresult") )
		std::cerr << THREADS << "\t" << GRIDS << "\t" << PHOTONS_IN_THREAD << "\t" << REPEAT << "\t" << gpu_time << "\t" << photonsperms << "\t" << cpu_time/gpu_time << "\t" << energy << "\t" << rho << "\t" << origin.x << "\t" << origin.y << "\t" << origin.z << "\t" << boxmin.x << "\t" << boxmin.y << "\t" << boxmin.z << "\t" << boxmax.x << "\t" << boxmax.y << "\t" << boxmax.z << std::endl;

}


////////////////////////////////////////////////////////////////////////////////
// Finalization: cleanup memory
////////////////////////////////////////////////////////////////////////////////
void finalize() {
	free( h_odata );
	free( h_csData );
	cutilSafeCall(cudaFree(d_odata));
	cutilSafeCall(cudaFreeArray(d_crossSection));
	cudaThreadExit();
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{
	printf("Welcome to Scintillator v1.0\nGPU based Monte Carlo simulation. (C) 2009 Attila Incze.\n");
	printf("------------------------------------\n");

	loadConfig(argc, argv);

	handleParams(argc, argv);

	initCalculation();

	simulationGPU();

	simulationCPU();

	printResults(argc, argv);

	finalize();

#ifdef WIN32
	cutilExit(argc, argv);
#endif
}

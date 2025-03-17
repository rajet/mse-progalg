#include <cassert>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include "Stopwatch.h"
#include "checkresult.h"

using Vector = std::vector<float>;

////////////////////////////////////////////////////////////////////////////////////////
// compiler directives
//#define _RANDOMPIVOT_ // random pivot chosing is too slow

#ifndef _RANDOMPIVOT_
////////////////////////////////////////////////////////////////////////////////////////
// determine median of a[p1], a[p2], and a[p3]
static int median(float a[], int p1, int p2, int p3) {
	const float ap1 = a[p1];
	const float ap2 = a[p2];
	const float ap3 = a[p3];

	if (ap1 <= ap2) {
		return (ap2 <= ap3) ? p2 : ((ap1 <= ap3) ? p3 : p1);
	} else {
		return (ap1 <= ap3) ? p1 : ((ap2 <= ap3) ? p3 : p2);
	}
}
#endif

////////////////////////////////////////////////////////////////////////////////////////
// serial quicksort
// sorts a[left]..a[right]
void quicksort(float a[], int left, int right) {
	// compute pivot
#ifdef _RANDOMPIVOT_
	std::default_random_engine e;
	std::uniform_int_distribution dist(left, right);

	const size_t pivotPos = dist(e);
#else
	const size_t pivotPos = median(a, left, left + (right - left)/2, right);
#endif
	const float pivot = a[pivotPos];

	int i = left, j = right;

	do {
		while (a[i] < pivot) i++;
		while (pivot < a[j]) j--;
		if (i <= j) {
			std::swap(a[i], a[j]);
			i++;
			j--;
		}
	} while (i <= j);
	if (left < j) quicksort(a, left, j);
	if (i < right) quicksort(a, i, right);
}

////////////////////////////////////////////////////////////////////////////////////////
// parallel quicksort
// sorts a[left]..a[right] using p threads 
void parallelQuicksort(float a[], int left, int right, int p) {
	assert(p > 0);
	assert(left >= 0 && left <= right);
	// TODO use OMP

	// Base case: use serial quicksort if array is small or only one thread available
	if (p <= 1 || (right - left) <= 1000) {
		quicksort(a, left, right);
		return;
	}
	
	// Compute pivot position using the median function
	const size_t pivotPos = median(a, left, left + (right - left)/2, right);
	const float pivot = a[pivotPos];
	
	// Partition the array like in serial quicksort
	int i = left, j = right;
	do {
		while (a[i] < pivot) i++;
		while (pivot < a[j]) j--;
		if (i <= j) {
			std::swap(a[i], a[j]);
			i++;
			j--;
		}
	} while (i <= j);
	
	// Split threads between partitions
	int leftThreads = p / 2;
	int rightThreads = p - leftThreads;
	
	// Initialize OpenMP parallel region if we're at the top level
	#pragma omp parallel num_threads(p) if(p > 1 && omp_get_thread_num() == 0)
	{
		#pragma omp single nowait
		{
			// Create tasks for recursive calls
			if (left < j) {
				#pragma omp task
				parallelQuicksort(a, left, j, leftThreads);
			}
			
			if (i < right) {
				#pragma omp task
				parallelQuicksort(a, i, right, rightThreads);
			}
			
			#pragma omp taskwait
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////
void quicksortTests(int n) {
	std::cout << "\nQuicksort Tests" << std::endl;
	Stopwatch sw;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;
	Vector data(n);
	Vector sortRef(n);
	Vector sort(n);

	// init arrays
	for (size_t i = 0; i < n; i++) sortRef[i] = data[i] = dist(e);
	const int p = omp_get_num_procs();

	// omp settings
	//omp_set_nested(true);
	omp_set_max_active_levels(30);
	std::cout << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "p = " << p << std::endl;
	std::cout << "Max Threads: " << omp_get_max_threads() << std::endl;
//	std::cout << "Nested Threads: " << boolalpha << (bool)omp_get_nested() << std::endl << std::endl;
	std::cout << "Nested Levels: " << omp_get_max_active_levels() << std::endl << std::endl;

	// stl sort
	sw.Start();
	std::sort(sortRef.begin(), sortRef.end());
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("std::sort:", sortRef.data(), sortRef.data(), ts, ts, n, p);

	// sequential quicksort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	quicksort(sort.data(), 0, n - 1);
	sw.Stop();
	check("sequential quicksort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel quicksort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	parallelQuicksort(sort.data(), 0, n - 1, p);
	sw.Stop();
	check("parallel quicksort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);
}
#include <cassert>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include "Stopwatch.h"
#include "checkresult.h"

using Vector = std::vector<float>;

///////////////////////////////////////////////////////////////////////////////
// Sequential Bitonic sort (used in performance tests)
// n must be a power of 2
static void bitonicSortSeq(float a[], const int n) {
	// compute d = log(n)
	int nn = n;
	int d = -1;

	while (nn) {
		d++;
		nn >>= 1;
	}

	int biti = 1;
	for (int i = 0; i < d; i++) {
		int bitj = biti; // bit j

		biti <<= 1; // bit i + 1
		for (int j = i; j >= 0; j--) {
			for (int k = 0; k < n; k++) {
				const int m = (~k & bitj) | (k & ~bitj); // xor

				if (m > k) {
					// only one of the two processing elements initiates the swap operation
					const bool bi = (k & biti) != 0;
					const bool bj = (k & bitj) != 0;
					if (bi == bj) {
						// comp_exchange_min on channel k with m
						// k takes the min, m the max
						if (a[k] > a[m]) std::swap(a[k], a[m]);
						
					} else {
						// comp_exchange_max on channel k with m
						// k takes the max, m the min
						if (a[k] < a[m]) std::swap(a[k], a[m]);
					}
				}
			}
			bitj >>= 1;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Bitonic sort implementation for p = n
// n must be a power of 2
// p parallel threads
static void bitonicSortOMP1(float a[], const int n, const int p) {
	// TODO use OMP
	
	// compute d = log(n)
	int nn = n;
	int d = -1;

	while (nn) {
		d++;
		nn >>= 1;
	}

	int biti = 1;
	for (int i = 0; i < d; i++) {
		int bitj = biti; // bit j

		biti <<= 1; // bit i + 1
		for (int j = i; j >= 0; j--) {
			#pragma omp for
			for (int k = 0; k < n; k++) {
				const int m = (~k & bitj) | (k & ~bitj); // xor

				if (m > k) {
					// only one of the two processing elements initiates the swap operation
					const bool bi = (k & biti) != 0;
					const bool bj = (k & bitj) != 0;
					if (bi == bj) {
						// comp_exchange_min on channel k with m
						// k takes the min, m the max
						if (a[k] > a[m]) std::swap(a[k], a[m]);
						
					} else {
						// comp_exchange_max on channel k with m
						// k takes the max, m the min
						if (a[k] < a[m]) std::swap(a[k], a[m]);
					}
				}
			}
			bitj >>= 1;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////
// Bitonic sort implementation for p < n, O(log^2(n))
// (used in performance tests)
// n and p must be a power of 2
// p parallel threads
void bitonicSortOMP2(float a[], const int n, const int p) {
	// TODO use OMP
	 // Compute d = log(n)
	 int nn = n;
	 int d = 0;
	 while (nn > 1) {
		 d++;
		 nn >>= 1;
	 }
 
	 // Set number of threads
	 omp_set_num_threads(p);
	 
	 int biti = 1;
	 for (int i = 0; i < d; i++) {
		 int bitj = biti; // bit j
		 
		 biti <<= 1; // bit i + 1
		 for (int j = i; j >= 0; j--) {
			 // Parallelize the outer loop across p threads
			 #pragma omp parallel
			 {
				 int tid = omp_get_thread_num();
				 int chunk_size = n / p;
				 int start = tid * chunk_size;
				 int end = (tid + 1) * chunk_size;
				 
				 // Each thread processes its chunk of the array
				 for (int k = start; k < end; k++) {
					 const int m = (~k & bitj) | (k & ~bitj); // xor
					 
					 if (m > k) {
						 // only one of the two processing elements initiates the swap operation
						 const bool bi = (k & biti) != 0;
						 const bool bj = (k & bitj) != 0;
						 if (bi == bj) {
							 // comp_exchange_min on channel k with m
							 // k takes the min, m the max
							 if (a[k] > a[m]) std::swap(a[k], a[m]);
						 } else {
							 // comp_exchange_max on channel k with m
							 // k takes the max, m the min
							 if (a[k] < a[m]) std::swap(a[k], a[m]);
						 }
					 }
				 }
			 } // Implicit barrier at the end of parallel region
			 
			 bitj >>= 1;
		 }
	 }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// compare-split of nlocal data elements
// input: a and b
// output: small and large
static void compareSplit(int nlocal, float a[], float b[], float small[], float large[]) {
	// TODO (use OMP sections)
    // Temporary arrays to hold merged data
    float* merged = new float[2 * nlocal];
    
    #pragma omp sections
    {
        #pragma omp section
        {
            // Merge the two sorted arrays
            int i = 0, j = 0, k = 0;
            while (i < nlocal && j < nlocal) {
                if (a[i] <= b[j]) {
                    merged[k++] = a[i++];
                } else {
                    merged[k++] = b[j++];
                }
            }
            
            // Copy any remaining elements
            while (i < nlocal) {
                merged[k++] = a[i++];
            }
            while (j < nlocal) {
                merged[k++] = b[j++];
            }
        }
    }
	
    
    // Divide the merged array into "small" and "large" parts
    #pragma omp sections
    {
        #pragma omp section
        {
            // Copy the smaller half to small[]
            for (int i = 0; i < nlocal; i++) {
                small[i] = merged[i];
            }
        }
        
        #pragma omp section
        {
            // Copy the larger half to large[]
            for (int i = 0; i < nlocal; i++) {
                large[i] = merged[nlocal + i];
            }
        }
    }
    
    delete[] merged;
}

///////////////////////////////////////////////////////////////////////////////
void bitonicsortTests(int n) {
	std::cout << "\nBitonic Sort Tests" << std::endl;
	Stopwatch sw;
	std::default_random_engine e;
	std::uniform_real_distribution<float> dist;
	Vector data(n);
	Vector sortRef(n);
	Vector sort(n);

	// init arrays
	for (int i = 0; i < n; i++) sortRef[i] = data[i] = dist(e);
	int p = omp_get_num_procs();

	// omp settings
	std::cout << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "p = " << p << std::endl;
	std::cout << "Max Threads: " << omp_get_max_threads() << std::endl;

	// stl sort
	sw.Start();
	std::sort(sortRef.begin(), sortRef.end());
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	check("std::sort:", sortRef.data(), sortRef.data(), ts, ts, n, p);

	// sequential bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	bitonicSortSeq(sort.data(), n);
	sw.Stop();
	check("sequential bitonic sort:", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	sw.Restart();
	bitonicSortOMP1(sort.data(), n, p);
	sw.Stop();
	check("parallel bitonic sort (p = n):", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);

	// parallel bitonic sort
	copy(data.begin(), data.end(), sort.begin());
	p = 8; assert(n%p == 0);
	sw.Restart();
	bitonicSortOMP2(sort.data(), n, p);
	sw.Stop();
	check("parallel bitonic sort (p < n):", sortRef.data(), sort.data(), ts, sw.GetElapsedTimeMilliseconds(), n, p);
}

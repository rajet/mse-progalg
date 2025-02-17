#include <sycl/sycl.hpp>

// SYCL reference manual
// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
// Windows Command Line
// icx-cl -O2 -EHsc -Qiopenmp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I..\Stopwatch main.cpp matrixmult.cpp vectoradd.cpp

//#define DEMO // uncomment this line if you want to run the Codeplay demo sample

#ifndef DEMO
void vectorAdditionTests();
void matrixMultiplicationTests();
#endif

static void demo() {
    // Creating buffer of 4 ints to be used inside the kernel code
    sycl::buffer<int, 1> buffer{ 40 };

    // Creating SYCL queue
    sycl::queue queue{};

    // show device name
	std::cout << std::endl << "SYCL on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Size of index space for kernel
    sycl::range<1> numOfWorkItems{ buffer.size() };

    // Submitting command group(work) to queue
    queue.submit([&](sycl::handler& cgh) {
        // Getting write only access to the buffer on a device
        auto bufferAcc = buffer.get_access<sycl::access::mode::write>(cgh);
        // Executing kernel
        cgh.parallel_for<class FillBuffer>(numOfWorkItems, [=](sycl::id<1> wi) {
            // Fill buffer with indexes
            bufferAcc[wi] = static_cast<int>(wi.get(0));
        });
    });

    // Getting read only access to the buffer on the host.
    // Implicit barrier waiting for queue to complete the work.
    auto hostAccessor = buffer.get_host_access();

    // Check the results
    bool mismatchFound{ false };

    for (size_t i = 0; i < buffer.size(); ++i) {
        if (hostAccessor[i] != i) {
            std::cout << "The result is incorrect for element: " << i
                << " , expected: " << i << " , got: " << hostAccessor[i]
                << std::endl;
            mismatchFound = true;
        }
    }

    if (!mismatchFound) {
        std::cout << "The results are correct!" << std::endl;
    }
}

int main() {
#ifdef DEMO
    demo();
#else
	vectorAdditionTests();
	matrixMultiplicationTests();	
#endif
}

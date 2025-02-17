#include <iostream>
#include <memory>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <sycl/sycl.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif
#include "FreeImagePlus.h"
#undef min

// SYCL reference manual
// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html

constexpr int BlockSize = 16;

////////////////////////////////////////////////////////////////////////
// compute and return distance 
static BYTE dist(int dx, int dy) {
	const int d = (int)sqrt(dx*dx + dy*dy);
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
// Sequential image processing (filtering).
// It computes the first derivative in x- and y-direction
void processSerial(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;

	// iterate all rows of the output image
	for (unsigned v = halfFilterSize; v < output.getHeight() - halfFilterSize; v++) {
		// iterate all pixels of the current row
		for (unsigned u = halfFilterSize; u < output.getWidth() - halfFilterSize; u++) {
			RGBQUAD inputColor;
			int horColor[3]{};
			int verColor[3]{};
			int filterPos = 0;

			// iterate all filter coefficients
			for (unsigned j = 0; j < filterSize; j++) {
				for (unsigned i = 0; i < filterSize; i++) {
					const int horFilterCoeff = horFilter[filterPos];
					const int verFilterCoeff = verFilter[filterPos];

					// read pixel color at position (u + i - halfFilterSize, v + j - halfFilterSize) and store the color in inColor
					input.getPixelColor(u + i - halfFilterSize, v + j - halfFilterSize, &inputColor);

					// apply one filter coefficient of the horitontal filter
					horColor[0] += horFilterCoeff*inputColor.rgbBlue;
					horColor[1] += horFilterCoeff*inputColor.rgbGreen;
					horColor[2] += horFilterCoeff*inputColor.rgbRed;
					// apply one filter coefficient of the vertical filter
					verColor[0] += verFilterCoeff*inputColor.rgbBlue;
					verColor[1] += verFilterCoeff*inputColor.rgbGreen;
					verColor[2] += verFilterCoeff*inputColor.rgbRed;

					filterPos++;
				}
			}

			// compute filter result and store it in output pixel (col, row)
			RGBQUAD outColor = { 
				dist(horColor[0], verColor[0]), 
				dist(horColor[1], verColor[1]), 
				dist(horColor[2], verColor[2]), 
				255 };
			output.setPixelColor(u, v, &outColor);
		}
	}
}

////////////////////////////////////////////////////////////////////////
// Optimized sequential image processing (filtering).
void processSerialOpt(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;

	// TODO
}

////////////////////////////////////////////////////////////////////////
// Parallel image processing (filtering) with OMP.
void processOMP(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	// TODO use OMP
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL.
void processSYCL(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltOffset(halfFilterSize, halfFilterSize);
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::range<2> procRange = imgRange - fltOffset;
	const sycl::nd_range<2> ndr(imgRange, {BlockSize, BlockSize});	// workgroup size: 16x16

	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
		sycl::accessor outAcc(outBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) {
			const sycl::id<2> outPos = ii.get_global_id();
			
			// TODO use SYCL
		});
	});
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL and nd_range.
// Use vectorization to process all three color channels in parallel.
void processSYCLvec(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltOffset(halfFilterSize, halfFilterSize);
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::range<2> procRange = imgRange - fltOffset;
	const sycl::nd_range<2> ndr(imgRange, {BlockSize, BlockSize});	// workgroup size: 16x16

	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
		sycl::accessor outAcc(outBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) {
			const sycl::id<2> outPos = ii.get_global_id();
			
			// TODO use SYCL and vectorization
		});
	});
}


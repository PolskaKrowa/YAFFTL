# yafftl
Yet Another Fast Fourier Transform library

---
## What is YAFFTL?

YAFFTL was born out of sheer desperation for not finding a good enough FFT library that was both easy to use, and very fast.

I wanted to be able to just run an `fft()` function, and be done with it. So i made YAFFTL to fill that void in open source software, as all the good FFT routines are either hidden away in closed-source software, have an uncountable amount of source files, or have thousands of functions to juggle around. When the average developer uses an FFT library, they expect it to do exactly what it says on the tin: Allow the user to perform forward/inverse FFT passes, or perform linear convolution.

YAFFTL uses AVX512 to perform operations on 8 bytes of data at a time, which significantly accelerates runtime on modern CPUs. However, I do understand that not all CPUs have AVX512 and are unable to get those performance improvements, So I've also included scalar branches which may possibly compile to AVX or AVX2 if compilers ever get proper code vectorisation sorted.

## Usage

Using YAFFTL is as easy as importing a header and calling some functions.

I'm being serious, just take a look:
```c++
#include "yafftl.hpp"
auto spectrum = yafftl::fft(signal);
auto back     = yafftl::ifft(spectrum);

// ideally, back will equal spectrum.

auto product  = yafftl::convolve(a, b);
```

## Licence

Apache 2.0, see [LICENCE](https://github.com/PolskaKrowa/YAFFTL/LICENCE) for details

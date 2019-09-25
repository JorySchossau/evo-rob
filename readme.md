### Requirements
* c++17
* LAPACK
* BLAS

### Install Armadillo Library
[see mac/win/nix instructions](https://solarianprogrammer.com/2017/03/24/getting-started-armadillo-cpp-linear-algebra-windows-mac-linux/)

### Compiling
c++ -larmadillo -std=c++17 -ffast-math -O3 -DNO_BOUNDS_CHECKING -o sci sci.cpp

### HPCC
This is undocumented, so icer had to tell me:
* clear spider cache
rm ~/.lmod.d/ -Rf
* load correct modules
ml armadillo

### The CLI Library (included)
I used [this file](https://github.com/CLIUtils/CLI11/releases/download/v1.8.0/CLI11.hpp) that I got from the [releases on this page](https://github.com/CLIUtils/CLI11/releases)

### The PrettyPrint (PPrint) Library (included)
I used [this file](https://raw.githubusercontent.com/p-ranav/pprint/master/include/pprint.hpp) I got on the [github page here](https://github.com/p-ranav/pprint)

### The Archive Serialization Library (included)
I used [this file](https://raw.githubusercontent.com/voidah/archive/master/archive.h) I got on the [github page here](https://github.com/voidah/archive)

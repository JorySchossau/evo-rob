### Requirements
* c++17
* LAPACK
* BLAS

### Install Armadillo Library
[see mac/win/nix instructions](https://solarianprogrammer.com/2017/03/24/getting-started-armadillo-cpp-linear-algebra-windows-mac-linux/)

### Get CLI Library
I used [this file](https://github.com/CLIUtils/CLI11/releases/download/v1.8.0/CLI11.hpp) that I got from the [releases on this page](https://github.com/CLIUtils/CLI11/releases)

### Get PrettyPrint (PPrint) Library
I used [this file](https://raw.githubusercontent.com/p-ranav/pprint/master/include/pprint.hpp) I got on the [github page here](https://github.com/p-ranav/pprint)

### Get Archive Serialization Library
I used [this file](https://raw.githubusercontent.com/voidah/archive/master/archive.h) I got on the [github page here](https://github.com/voidah/archive)

### Compiling
c++ -larmadillo -std=c++17 -ffast-math -O3 -DNO_BOUNDS_CHECKING -o sci sci.cpp

### HPCC
This is undocumented, so icer had to tell me:
* clear spider cache
rm ~/.lmod.d/ -Rf
* load correct modules
ml armadillo


mexCMD=/usr/local/MATLAB/R2014b/bin/mex
all: resize features cascade fconv_var_dim


resize: resize.cc
	${mexCMD} -O \
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	resize.cc

features: features.cc
	${mexCMD} -O \
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	features.cc	

cascade: cascade.cc model.cc model.h
	${mexCMD} -O 						\
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	cascade.cc model.cc

fconv_var_dim: fconv_var_dim.cc
	${mexCMD} -O				\
	CXXOPTIMFLAGS="-O3 -DNDEBUG -fomit-frame-pointer"  \
	LDOPTIMFLAGS="-O3"						\
	CXXFLAGS="\$$CXXFLAGS -Wall"	\
	LDFLAGS="\$$LDFLAGS -Wall"		\
	fconv_var_dim.cc

clean:
	rm *.mex*

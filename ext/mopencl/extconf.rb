require "mkmf"

$LDFLAGS += " -framework OpenCL"
create_makefile("mopencl/mopencl")

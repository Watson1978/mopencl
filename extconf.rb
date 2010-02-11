require "mkmf"

$LDFLAGS += " -framework OpenCL"
create_makefile("OpenCLBase")

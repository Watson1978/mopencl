# -*- coding: utf-8 -*-
#/usr/local/bin/macruby
require "opencl"
require "benchmark"

opencl = OpenCL.new
# When true is set in use_cpu, the program is executed on CPU.
opencl.use_cpu = true

opencl.program <<EOF
  __kernel sum(__global float *in, __global int *out, int total) {
    int i = get_global_id(0);
    if (i < total) out[i] = ((float)in[i] + 0.5) / 3.8 + 2.0;
  }
EOF

DATA_SIZE = 3333333

data = (1..DATA_SIZE).to_a

input  = opencl.set_input(data, DATA_SIZE)
output = opencl.set_output(DATA_SIZE)

opencl.sum(input, output, DATA_SIZE)

Benchmark.bmbm do |x|
  x.report("map        ") { data.map {|x| (x.to_f + 0.5) / 3.8 + 2.0 } }
  x.report("OpenCL(cpu)") { opencl.sum(input, output, DATA_SIZE) }
end
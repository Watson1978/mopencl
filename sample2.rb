# -*- coding: utf-8 -*-
#/usr/local/bin/macruby
require "opencl"

opencl = OpenCL.new
# When true is set in use_cpu, the program is executed on CPU.
#opencl.use_cpu = true

opencl.program <<EOF
// OpenCL kernel that computes the square of an input array
__kernel square(
__global float* input,
__global float* output,
__global int count)
{
int i = get_global_id(0);
if(i < count)
output[i] = input[i] * input[i];
}
EOF

input  = opencl.set_input((1..32768).to_a, 32768)
output = opencl.set_output(32768)

opencl.square(input, output, 32768)
p output.result.last



#/usr/local/bin/macruby
# http://blog.0x82.com/2010/1/23/opencl-in-macruby-hack-not-very-useful
require "OpenCLBase"

OpenCLBase.devices.each do |device|
  info = device.info
  puts "Trying on #{info[:name]} #{info[:vendor]} #{info[:version]} [#{info[:type]}]"
 
  context = device.create_context
  queue = context.create_command_queue
  program = context.create_program <<EOF
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
  
  kernel = program.create_kernel("square")
  
  input = context.create_read_buffer(32768)
  output = context.create_write_buffer(32768)
  
  input.write(queue, (1..32768).to_a)
  
  kernel.set_arg(0, input)
  kernel.set_arg(1, output)
  kernel.set_arg(2, 32768)
  
  kernel.enqueue_nd_range(device, queue, 32768)
  puts output.read(queue).last
end


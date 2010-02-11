# -*- coding: utf-8 -*-
require "OpenCLBase"

class OpenCL
#   def initialize()
#     @cpu = OpenCLBase.get_cpu
#     @gpu = OpenCLBase.get_gpu
#   end

  def info()
    info = []
    OpenCLBase.devices.each do |device|
      info << device.info
    end
    return info
  end

  def able_gpu?()
    @gpu ||= OpenCLBase.get_gpu
    return true if(@gpu != nil)
    return false
  end

  def get_device()
    @gpu ||= OpenCLBase.get_gpu
    return @gpu if(self.able_gpu?())

    @cpu ||= OpenCLBase.get_cpu
    return @cpu
  end

  def program(prog)
    @device = self.get_device()

    @context = @device.create_context()
    @queue   = @context.create_command_queue()
    @program = @context.create_program(prog)
    return @program
  end

  def set_input(val, size)
    @input = @context.create_read_buffer(size)
    @input.write(@queue, val)
    return @input
  end

  def set_output(size)
    @output = @context.create_write_buffer(size)
    @output_size = size
    return @output
  end

  def method_missing(method_name, *args)
    method = method_name.to_s
    @kernel = @program.create_kernel(method);
    args.each_with_index do |arg, index|
      @kernel.set_arg(index, arg)
    end

    @kernel.enqueue_nd_range(@device, @queue, @output_size)

    @result = @output.read(@queue)
    return @result
  end

end

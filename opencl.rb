# -*- coding: utf-8 -*-
require "OpenCLBase"

class OpenCL
  # When true is set in use_cpu, the program is executed on CPU.
  attr_accessor :use_cpu

  def initialize()
    self.use_cpu = false
  end

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
    return @gpu if(!self.use_cpu and self.able_gpu?())

    @cpu ||= OpenCLBase.get_cpu
    return @cpu
  end

  def program(source)
    @source = source
    if(@source =~ /([^\(]+)\(/m)
      @source = $1
    end
    @device = self.get_device()

    @context = @device.create_context()
    @queue   = @context.create_command_queue()
    @program = @context.create_program(source)
    return @program
  end

  def set_input(val, size)
    @work_size ||= size
    @work_size = size if(@work_size < size)
    @input = @context.create_read_buffer(size)
    @input.write(@queue, val)
    return @input
  end

  def set_output(size)
    @work_size ||= size
    @work_size = size if(@work_size < size)
    return OutputBuffer.new(@context, @queue, size)
  end

  def method_missing(method_name, *args)
    method = method_name.to_s

    if(@source =~ /#{method}/m)
      @kernel = @program.create_kernel(method);
      args.each_with_index do |arg, index|
        case arg
        when OutputBuffer
          arg = arg.output
        end

        @kernel.set_arg(index, arg)
      end

      @kernel.enqueue_nd_range(@device, @queue, @work_size)
    else
      raise NameError.new("Undefined method : #{method}")
    end
  end

end

class OutputBuffer
  attr_accessor :output

  def initialize(context, queue, size)
    @queue = queue
    self.output = context.create_write_buffer(size)
    return self
  end

  def result()
    return @output.read(@queue)
  end
end

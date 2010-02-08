#include "ruby.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

typedef struct {
  struct RBasic basic;
  cl_mem buffer;
  cl_uint size;
} rb_opencl_buffer_t;
#define ROpenCLBuffer(val) ((rb_opencl_buffer_t*)val)

typedef struct {
  struct RBasic basic;
  cl_kernel kernel;
  VALUE name;
} rb_opencl_kernel_t;
#define ROpenCLKernel(val) ((rb_opencl_kernel_t*)val)

typedef struct {
  struct RBasic basic;
  cl_program program;
} rb_opencl_program_t;
#define ROpenCLProgram(val) ((rb_opencl_program_t*)val)

typedef struct {
  struct RBasic basic;
  cl_command_queue queue;
} rb_opencl_command_queue_t;
#define ROpenCLCommandQueue(val) ((rb_opencl_command_queue_t*)val)

typedef struct {
  struct RBasic basic;
  cl_device_id device_id;
} rb_opencl_device_t;
#define ROpenCLDevice(val) ((rb_opencl_device_t*)val)

typedef struct {
  struct RBasic basic;
  cl_context context;
  cl_device_id device_id;
} rb_opencl_context_t;
#define ROpenCLContext(val) ((rb_opencl_context_t*)val)

static VALUE mOpenCL;
static VALUE cDevice;
static VALUE cContext;
static VALUE cCommandQueue;
static VALUE cProgram;
static VALUE cKernel;
static VALUE cBuffer;

// Buffer methods
static VALUE
rb_buffer_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_buffer_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_buffer_init(VALUE self, SEL sel, VALUE context, VALUE size, VALUE read_only)
{
  ROpenCLBuffer(self)->size  = NUM2UINT(size);

  int err;
  if(read_only == Qfalse)
    ROpenCLBuffer(self)->buffer = clCreateBuffer((cl_context) context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
  else
    ROpenCLBuffer(self)->buffer = clCreateBuffer((cl_context) context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);

  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  rb_ivar_set(self, rb_intern("buffer"), (VALUE)ROpenCLBuffer(self)->buffer);

  return self;
}

static VALUE
rb_buffer_write(VALUE self, SEL sel, VALUE queue, VALUE data)
{
  int i;
  cl_int err;
  cl_command_queue q = (cl_command_queue) rb_ivar_get(queue, rb_intern("queue"));
  float* d = xmalloc2(ROpenCLBuffer(self)->size, sizeof(float));

  for(i=0; i<ROpenCLBuffer(self)->size; i++) {
    d[i] = NUM2DBL(rb_ary_entry(data, i));
  }

  err = clEnqueueWriteBuffer(q, ROpenCLBuffer(self)->buffer, CL_TRUE, 0, sizeof(double)*ROpenCLBuffer(self)->size, d, 0, NULL, NULL);

  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  return Qnil;
}

static VALUE
rb_buffer_read(VALUE self, SEL sel, VALUE queue)
{
  int i;
  cl_int err;
  cl_command_queue q = (cl_command_queue) rb_ivar_get(queue, rb_intern("queue"));
  float* d = xmalloc2(ROpenCLBuffer(self)->size, sizeof(float));

  err = clEnqueueReadBuffer(q, ROpenCLBuffer(self)->buffer, CL_TRUE, 0, sizeof(float)*ROpenCLBuffer(self)->size, d, 0, NULL, NULL);
  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  VALUE res = rb_ary_new2(ROpenCLBuffer(self)->size);
  for(i=0; i<ROpenCLBuffer(self)->size; i++) {
    rb_ary_push(res, rb_float_new(d[i]));
  }
  return res;
}

// Kernel methods
static VALUE
rb_kernel_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_kernel_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_kernel_init(VALUE self, SEL sel, VALUE program, VALUE name)
{
  int err;
 
  ROpenCLKernel(self)->kernel = clCreateKernel(ROpenCLProgram(program)->program, StringValuePtr(name), &err);
  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  return self;
}

static VALUE
rb_kernel_set_arg(VALUE self, SEL sel, VALUE idx, VALUE obj)
{
  int err;
  cl_mem buffer;
  unsigned int fixnum;

  switch(TYPE(obj)) {
    case T_NATIVE:
      buffer = (cl_mem) rb_ivar_get(obj, rb_intern("buffer"));
      err = clSetKernelArg(ROpenCLKernel(self)->kernel, NUM2UINT(idx), sizeof(cl_mem), &buffer);
      break;

    case T_FIXNUM:
      fixnum = NUM2UINT(obj);
      err = clSetKernelArg(ROpenCLKernel(self)->kernel, NUM2UINT(idx), sizeof(unsigned int), &fixnum);
      break;

    default:
      rb_raise(rb_eTypeError, "not valid type");
      break;
  }

  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  return Qnil;
}

static VALUE
rb_kernel_enqueue(VALUE self, SEL sel, VALUE device, VALUE queue, VALUE count)
{
  int err;
  size_t local;
  cl_device_id d = (cl_device_id) rb_ivar_get(device, rb_intern("device_id"));

  err = clGetKernelWorkGroupInfo(ROpenCLKernel(self)->kernel, d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
  if(err != CL_SUCCESS)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  size_t c = NUM2UINT(count);
  if(local > c)
    local = c;

  cl_command_queue q = (cl_command_queue)rb_ivar_get(queue, rb_intern("queue"));
  err = clEnqueueNDRangeKernel(
           q,
           ROpenCLKernel(self)->kernel,
           1, NULL, &c, &local, 0, NULL, NULL);
  if(err)
    rb_raise(rb_eArgError, "Failed with error `%d'", err);

  clFinish(q);
  return Qnil;
}

// Program methods
static VALUE
rb_program_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_program_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_program_init(VALUE self, SEL sel, VALUE program)
{
  ROpenCLProgram(self)->program = (cl_program) program;
  return self;
}

static VALUE
rb_program_create_kernel(VALUE self, SEL sel, VALUE name)
{
  VALUE res = rb_kernel_alloc(cKernel, 0);
  rb_kernel_init(res, 0, self, name);
  return res;
}

// Command queue methods
static VALUE
rb_command_queue_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_command_queue_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_command_queue_init(VALUE self, SEL sel, VALUE device_id, VALUE context)
{
  int err;
  ROpenCLCommandQueue(self)->queue = clCreateCommandQueue((cl_context)context, (cl_device_id)device_id, 0, &err);
  if(err != CL_SUCCESS) {
    rb_raise(rb_eArgError, "Failed with error `%d'", err);
  }

  rb_ivar_set(self, rb_intern("queue"), (VALUE)ROpenCLCommandQueue(self)->queue);

  return self;
}

// Context methods
static VALUE
rb_context_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_context_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_context_init(VALUE self, SEL sel, VALUE value)
{
  int err;
  ROpenCLContext(self)->context = clCreateContext(0, 1, (cl_device_id*) &value, NULL, NULL, &err);
  if(err != CL_SUCCESS) {
    rb_raise(rb_eArgError, "Failed with error `%d'", err);
  }
  ROpenCLContext(self)->device_id = (cl_device_id)value;

  return self;
}

static VALUE
rb_context_create_command_queue(VALUE self, SEL sel)
{
  VALUE res = rb_command_queue_alloc(cCommandQueue, 0);
  rb_command_queue_init(res, 0, (VALUE)ROpenCLContext(self)->device_id, (VALUE)ROpenCLContext(self)->context);
  return res;
}

static VALUE
rb_context_create_program(VALUE self, SEL sel, VALUE kernel)
{
  int err;
  cl_program program;
  char *source = StringValuePtr(kernel);

  program = clCreateProgramWithSource(ROpenCLContext(self)->context, 1, (const char **)&source, NULL, &err);
  if(err != CL_SUCCESS) {
    rb_raise(rb_eArgError, "Failed with error `%d'", err);
  }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    clGetProgramBuildInfo(program, ROpenCLContext(self)->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    rb_raise(rb_eArgError, "Failed with error `%s'", buffer);
  }

  VALUE res = rb_program_alloc(cProgram, 0);
  rb_program_init(res, 0, (VALUE)program);
  return res;
}

static VALUE
rb_context_create_read_buffer(VALUE self, SEL sel, VALUE size)
{
  VALUE res = rb_buffer_alloc(cBuffer, 0);
  rb_buffer_init(res, 0, (VALUE)ROpenCLContext(self)->context, size, Qtrue);
  return res;
}

static VALUE
rb_context_create_write_buffer(VALUE self, SEL sel, VALUE size)
{
  VALUE res = rb_buffer_alloc(cBuffer, 0);
  rb_buffer_init(res, 0, (VALUE)ROpenCLContext(self)->context, size, Qfalse);
  return res;
}

// Device methods
static VALUE
rb_device_alloc(VALUE klass, SEL sel)
{
  NEWOBJ(s, rb_opencl_device_t);
  OBJSETUP(s, klass, RUBY_T_NATIVE);
  return (VALUE)s;
}

static VALUE
rb_device_init(VALUE self, SEL sel, VALUE value)
{
  ROpenCLDevice(self)->device_id = (cl_device_id)value;
  rb_ivar_set(self, rb_intern("device_id"), (VALUE)ROpenCLDevice(self)->device_id);
  return self;
}

static VALUE
rb_device_create_context(VALUE self, SEL sel)
{
  VALUE res = rb_context_alloc(cContext, 0);
  rb_context_init(res, 0, (VALUE) ROpenCLDevice(self)->device_id);
  return res;
}

static VALUE
rb_device_info(VALUE self, SEL sel)
{
  char device_string[1024];
  cl_device_type type;
  cl_uint compute_units;
  size_t working_dims;
  size_t workitem_size[3];
  size_t workgroup_size;
  cl_uint clock_frequency;
  cl_device_id device_id = ROpenCLDevice(self)->device_id;
  VALUE hash = rb_hash_new();

  // CL_DEVICE_NAME
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("name")), rb_str_new2(device_string));

  // CL_DEVICE_VENDOR
  clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("vendor")), rb_str_new2(device_string));

  // CL_DEVICE_VERSION
  clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("version")), rb_str_new2(device_string));

  // CL_DEVICE_TYPE
  clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if(type & CL_DEVICE_TYPE_CPU)
    strcpy(device_string, "cpu");
  else if(type & CL_DEVICE_TYPE_GPU)
    strcpy(device_string, "gpu");
  else if(type & CL_DEVICE_TYPE_ACCELERATOR)
    strcpy(device_string, "accelerator");
  else if(type & CL_DEVICE_TYPE_DEFAULT)
    strcpy(device_string, "default");
  else
    strcpy(device_string, "unknown");
  rb_hash_aset(hash, ID2SYM(rb_intern("type")), ID2SYM(rb_intern(device_string)));

  // CL_DEVICE_MAX_COMPUTE_UNITS
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_compute_units")), UINT2NUM(compute_units));

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(working_dims), &working_dims, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_work_item_dimensions")), UINT2NUM(working_dims));

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  VALUE dimensions = rb_ary_new();
  rb_ary_push(dimensions, UINT2NUM(workitem_size[0]));
  rb_ary_push(dimensions, UINT2NUM(workitem_size[1]));
  rb_ary_push(dimensions, UINT2NUM(workitem_size[2]));
  rb_hash_aset(hash, ID2SYM(rb_intern("max_work_item_sizes")), dimensions);

  // CL_DEVICE_WORK_GROUP_SIZE
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_work_group_size")), UINT2NUM(workgroup_size));

  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_clock_frequency")), UINT2NUM(clock_frequency));

  // CL_DEVICE_ADDRESS_BITS
  cl_uint addr_bits;
  clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("address_bits")), UINT2NUM(addr_bits));

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  cl_ulong max_mem_alloc_size;
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_mem_alloc_size")), UINT2NUM((unsigned int)(max_mem_alloc_size / (1024 * 1024))));

  // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_ulong mem_size;
  clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("global_mem_size")), UINT2NUM((unsigned int)(mem_size / (1024 * 1024))));

  // CL_DEVICE_ERROR_CORRECTION_SUPPORT
  cl_bool error_correction_support;
  clGetDeviceInfo(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("error_correction_support")), error_correction_support == CL_TRUE ? Qtrue : Qfalse);

  // CL_DEVICE_LOCAL_MEM_TYPE
  cl_device_local_mem_type local_mem_type;
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("local_mem_type")), local_mem_type == 1 ? rb_str_new2("local") : rb_str_new2("global"));

  // CL_DEVICE_LOCAL_MEM_SIZE
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("local_mem_size")), UINT2NUM((unsigned int)(mem_size / 1024)));

  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_constant_buffer_size")), UINT2NUM((unsigned int)(mem_size / 1024)));

  // CL_DEVICE_QUEUE_PROPERTIES
  cl_command_queue_properties queue_properties;
  clGetDeviceInfo(device_id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
  VALUE properties = rb_ary_new();
  if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
    rb_ary_push(properties, ID2SYM(rb_intern("out_of_order_execution_mode_enable")));
  if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
    rb_ary_push(properties, ID2SYM(rb_intern("profiling_enable")));
  rb_hash_aset(hash, ID2SYM(rb_intern("queue_properties")), properties);

  // CL_DEVICE_IMAGE_SUPPORT
  cl_bool image_support;
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image_support")), image_support == CL_TRUE ? Qtrue : Qfalse);

  // CL_DEVICE_MAX_READ_IMAGE_ARGS
  cl_uint max_read_image_args;
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_read_image_args")), UINT2NUM(max_read_image_args));

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint max_write_image_args;
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("max_write_image_args")), UINT2NUM(max_write_image_args));
  
  // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
  size_t szMaxDims[5];
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image2d_max_width")), UINT2NUM(szMaxDims[0]));
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image2d_max_height")), UINT2NUM(szMaxDims[1]));
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image3d_max_width")), UINT2NUM(szMaxDims[2]));
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image3d_max_height")), UINT2NUM(szMaxDims[3]));
  clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
  rb_hash_aset(hash, ID2SYM(rb_intern("image3d_max_depth")), UINT2NUM(szMaxDims[4]));
  
  // CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the string onto separate lines
  clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_string), &device_string, NULL);
  if (strcmp(device_string, "")) {
    rb_hash_aset(hash, ID2SYM(rb_intern("extensions")), rb_str_new2(device_string));
  }

  return hash;
}

// OpenCL methods
static VALUE
rb_opencl_devices(VALUE klass, SEL sel)
{
  int err;
  int i;
  cl_device_id* device_ids;
  cl_uint number_of_devices;

  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &number_of_devices);
  if(err != CL_SUCCESS) {
    rb_raise(rb_eArgError, "Failed with error `%d'", err);
  }

  device_ids = (cl_device_id*) xmalloc(number_of_devices);
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, number_of_devices, device_ids, NULL);

  VALUE res = rb_ary_new();
  for(i=0; i<number_of_devices; i++) {
    VALUE o = rb_device_alloc(cDevice, 0);
    rb_device_init(o, 0, (VALUE) device_ids[i]);
    rb_ary_push(res, o);
  }

  return res;
}

void
Init_OpenCL(void)
{
  mOpenCL = rb_define_module("OpenCL");
  rb_objc_define_module_function(mOpenCL, "devices", rb_opencl_devices, 0);

  cDevice = rb_define_class_under(mOpenCL, "Device", rb_cObject);
  rb_objc_define_method(*(VALUE*)cDevice, "alloc", rb_device_alloc, 0);
  rb_objc_define_method(cDevice, "initialize", rb_device_init, 1);
  rb_objc_define_method(cDevice, "info", rb_device_info, 0);
  rb_objc_define_method(cDevice, "create_context", rb_device_create_context, 0);

  cContext = rb_define_class_under(mOpenCL, "Context", rb_cObject);
  rb_objc_define_method(*(VALUE*)cContext, "alloc", rb_context_alloc, 0);
  rb_objc_define_method(cContext, "initialize", rb_context_init, 1);
  rb_objc_define_method(cContext, "create_command_queue", rb_context_create_command_queue, 0);
  rb_objc_define_method(cContext, "create_program", rb_context_create_program, 1);
  rb_objc_define_method(cContext, "create_read_buffer", rb_context_create_read_buffer, 1);
  rb_objc_define_method(cContext, "create_write_buffer", rb_context_create_write_buffer, 1);

  cCommandQueue = rb_define_class_under(mOpenCL, "CommandQueue", rb_cObject);
  rb_objc_define_method(*(VALUE*)cCommandQueue, "alloc", rb_command_queue_alloc, 0);
  rb_objc_define_method(cCommandQueue, "initialize", rb_command_queue_init, 2);

  cProgram = rb_define_class_under(mOpenCL, "Program", rb_cObject);
  rb_objc_define_method(*(VALUE*)cProgram, "alloc", rb_program_alloc, 0);
  rb_objc_define_method(cProgram, "initialize", rb_program_init, 1);
  rb_objc_define_method(cProgram, "create_kernel", rb_program_create_kernel, 1);

  cKernel = rb_define_class_under(mOpenCL, "Kernel", rb_cObject);
  rb_objc_define_method(*(VALUE*)cKernel, "alloc", rb_kernel_alloc, 0);
  rb_objc_define_method(cKernel, "initialize", rb_kernel_init, 2);
  rb_objc_define_method(cKernel, "set_arg", rb_kernel_set_arg, 2);
  rb_objc_define_method(cKernel, "enqueue_nd_range", rb_kernel_enqueue, 3);

  cBuffer = rb_define_class_under(mOpenCL, "Buffer", rb_cObject);
  rb_objc_define_method(*(VALUE*)cBuffer, "alloc", rb_buffer_alloc, 0);
  rb_objc_define_method(cBuffer, "initialize", rb_buffer_init, 3);
  rb_objc_define_method(cBuffer, "write", rb_buffer_write, 2);
  rb_objc_define_method(cBuffer, "read", rb_buffer_read, 1);
}

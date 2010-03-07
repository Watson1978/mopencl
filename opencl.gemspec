require 'rubygems'
 
spec = Gem::Specification.new do |s|
  
  s.name = 'opencl'
  s.version = "0.0.1"
  s.platform = Gem::Platform::RUBY
  s.summary = "OpenCL library for MacRuby"
  s.description = "This is a OpenCL library for MacRuby."
  s.files = Dir.glob("{lib,test,bin}/**/*") + ['Rakefile']
  s.require_path = 'lib'
  s.extensions << 'Rakefile'
  s.author = "Watson"
  s.email = "watson1978@gmail.com"
  s.rubyforge_project = "opencl"
  s.homepage = "http://github.com/Watson1978/opencl"
 
end
 
if $0==__FILE__
  Gem::Builder.new(spec).build
end

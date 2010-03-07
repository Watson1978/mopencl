#!/usr/bin/env rake
MACRUBY = "/usr/local/bin/macruby"

task :default => :build do ; end

task :build do
  sh "cd lib; #{MACRUBY} extconf.rb"
  sh "cd lib; make"
end

task :pkg do
  sh "gem build opencl.gemspec"
end

task :clean do
  sh "rm *.gem"
  sh "cd lib; make clean"
  sh "cd lib; rm Makefile"
end

MACRUBY = "/usr/local/bin/macruby"

task :default do
  sh "#{MACRUBY} extconf.rb"
  sh "make"
end

task :clean do
  sh "make clean"
  sh "rm Makefile"
end

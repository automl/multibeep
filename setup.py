from distutils.core import setup, Extension
import distutils.command.build
import numpy as np
from subprocess import call
from Cython.Build import cythonize



include_dirs = ['./include', np.get_include(),'.', './multibeep/', './lib/']
#extra_compile_args = ['-O2', '-std=c++11']
extra_compile_args = ['-O0', '-g', '-std=c++11', '-Wall']









extensions = cythonize( list(
		map(lambda t:Extension(	t[0],
							sources=t[1],
							language="c++",
							include_dirs=include_dirs,
							extra_compile_args = extra_compile_args
						),
				[	('multibeep.util',		['multibeep/util.pyx']),
					('multibeep.arms',		['multibeep/arms.pyx']),
					('multibeep.bandits',	['multibeep/bandits.pyx']),
					('multibeep.policies',	['multibeep/policies.pyx']),
				]

		)), compiler_directives={'embedsignature':True})




setup(
	name='multibeep',
	version='0.0.1',
	author='Joel Kaiser, Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 2 - Pre alpha'],
	packages=['multibeep'],
	ext_modules=extensions
)

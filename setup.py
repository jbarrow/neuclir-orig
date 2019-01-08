from distutils.core import setup

setup(
    name='NeuCLIR',
    version='0.1dev',
    packages=['neuclir','neuclir.readers','neuclir.models','neuclir.datasets','neuclir.metrics'],
    license='Academic License',
    long_description=open('README.md').read(),
)

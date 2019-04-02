import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='executable_file',
     version='0.1',
     scripts=['executable_file'] ,
     author="Petros Christodoulou",
     author_email="p.christodoulou2@gmail.com",
     description="Build neural networks in 1 line",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/p-christ/nn_builder",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
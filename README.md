# Py2CUDA
Convert python code to CUDA.

## Usage
To convert a python file say named `py_file.py` to CUDA, run `python generate_cuda.py --file py_file.py --arch {your_gpu_arch}`, (if `--arch` is not specified, `sm_61` will be used).  
Example run: `python generate_cuda.py --file test.py --arch sm_86`
# Py2CUDA
Convert python code to CUDA.

## Usage
To convert a python file say named `py_file.py` to CUDA, run `python generate_cuda.py --file py_file.py --arch {your_gpu_arch}`, (if `--arch` is not specified, `sm_61` will be used). Output will be saved in `./compiled`.  
Example run: `python generate_cuda.py --file ./examples/test.py --arch sm_86`
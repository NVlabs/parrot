
## kp (Kernel Profiler)

**kp** is a simple tool that wraps `nsys` to create a nice plot comparing CUDA kernels run across different python files.

![](assets/sample.png)
```
usage: kp [-h] [-p] [-v] [-f FILTER] [-e] subdir

Run nsys profiling and generate plots.

positional arguments:
  subdir                name of the subdirectory containing python files (e.g. sf2, mgc).

options:
  -h, --help            show this help message and exit
  -p, --plot-only       skip profiling and only generate plots
  -v, --verbose         enable verbose output during profiling
  -f FILTER, --filter FILTER
                        only profile Python files containing this string
  -e, --export-html     export plot as HTML file
```
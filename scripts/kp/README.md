
## kp (Kernel Profiler)

**kp** is a simple tool that wraps `nsys` to create a nice plot comparing CUDA kernels run across different python files.

![](assets/sample.png)
```
usage: kp [-h] [-p] [-v] [-f FILTER] [-e] [-n LOWER UPPER] subdir

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
  -n LOWER UPPER, --sweep LOWER UPPER
                        sweep N over geometric 10x range and produce a line chart (e.g. -n 10K 10M)
```

### Sweep mode

Use `-n` to profile across a range of input sizes and generate a line chart (one line per implementation):

```
kp -n 10K 10M max_abs_diff2
```

This runs each file with N = 10K, 100K, 1M, 10M and plots total kernel time vs N.
Profiled files must accept N as a command-line argument (falls back to their default when omitted).
Results are saved to `nsys_results/sweep_results.csv` so `-p` can re-plot without re-profiling.
# libsmctrl: Quick & Easy Hardware Compute Partitioning on NVIDIA GPUs

This library was developed as part of the following paper:

_J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium, pp. 54-66, May 2023._

Please cite this paper in any work which leverages our library. Here's the BibTeX entry:
```
@inproceedings{bakita2023hardware,
  title={Hardware Compute Partitioning on {NVIDIA} {GPUs}},
  author={Bakita, Joshua and Anderson, James H},
  booktitle={Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium},
  year={2023},
  month={May},
  pages={54--66},
  _series={RTAS}
}
```

Please see [the paper](https://www.cs.unc.edu/~jbakita/rtas23.pdf) and libsmctrl.h for details and examples of how to use this library.
We strongly encourage consulting those resources first; the below comments serve merely as an appendum.

## Run Tests
To test partitioning:
```
make tests
./libsmctrl_test_global_mask
./libsmctrl_test_stream_mask
./libsmctrl_test_next_mask
```

And if `nvdebug` has been installed:
```
./libsmctrl_test_gpu_info
```

## Supported GPUs

#### Known Working

- NVIDIA GPUs from compute capability 3.5 through 8.9, including embedded "Jetson" GPUs
- CUDA 8.1 through 12.2
- `x86_64` and Jetson `aarch64` platforms

#### Known Issues

- `next_mask` will not override `stream_mask` on CUDA 12.0+
- `global_mask` and `next_mask` cannot disable TPCs with IDs above 128
    - Only relevant on GPUs with over 128 TPCs, such as the RTX 6000 Ada
- Untested on H100 (compute capability 9.0)
- Untested on non-Jetson `aarch64` platforms

## Important Limitations

1. Only supports partitioning _within_ a single GPU context.
   At time of writing, it is challenging to impossible to share a GPU context across multiple CPU address spaces.
   The implication is that your applications must first be combined together into a single CPU process.
2. No aspect of this system prevents implicit synchronization on the GPU.
   See prior work, particularly that of Amert et al. (perhaps the CUPiD^RT paper), for ways to avoid this.

## Porting to New Architectures

Build the tests with `make tests`. And then run the following:
```
for (( i=0; $?!=0; i+=8 )); do MASK_OFF=$i ./libsmctrl_test_stream_mask; done
```

How this works:

1. If `MASK_OFF` is set, `libsmctrl` applies this as a byte offset to a base address for the location
   of the SM mask fields in CUDA's stream data structure.
  - That base address is the one for CUDA 12.2 at time of writing
2. The stream masking test is run.
3. If the test succeeded (returned zero) the loop aborts, otherwise it increments the offset to attempt and repeats.

Once this loop aborts, take the found offset and add it into the switch statement for the appropriate CUDA version and CPU architecture.

## TODO

- Add a test to check that more-granularly-set compute masks override more-corsely-set ones.

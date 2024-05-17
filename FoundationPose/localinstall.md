# RTX 4090 support

```
https://github.com/NVlabs/FoundationPose/issues/27
```

# RealSense
```
https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
```

# Mismatch cuda version
```
nvcc --version # Check the cuda toolkit version cuda>=12.1
# Download the Cuda toolkit following: https://developer.nvidia.com/cuda-toolkit-archive
# Or find the installed cuda version under /usr/local
export PATH="/usr/local/cuda-12.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
```
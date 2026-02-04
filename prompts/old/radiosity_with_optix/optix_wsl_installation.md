# OptiX Installation Guide for WSL2

## Prerequisites Check

### 1. Verify WSL2 is Running
```bash
wsl --version
# Should show WSL version 2.x.x
```

If not on WSL2:
```bash
wsl --set-version Ubuntu 2
```

### 2. Check GPU Access
```bash
nvidia-smi
```

**Expected output**: NVIDIA driver version and GPU information  
**If this fails**: Install NVIDIA drivers on Windows host from [NVIDIA WSL Page](https://developer.nvidia.com/cuda/wsl)

## Step 1: Install CUDA Toolkit

### Remove Old CUDA (if exists)
```bash
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo apt-get autoremove
sudo apt-get autoclean
```

### Install CUDA 12.3 (or latest)
```bash
# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get -y install cuda-toolkit-12-3

# Verify installation
/usr/local/cuda/bin/nvcc --version
```

### Set Environment Variables
Add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Apply changes:
```bash
source ~/.bashrc
```

## Step 2: Download OptiX SDK

### Get OptiX from NVIDIA
1. Go to: https://developer.nvidia.com/designworks/optix/download
2. **NVIDIA Developer Account Required** (free registration)
3. Download: **OptiX 7.7.0** or later (Linux 64-bit)
4. File will be named something like: `NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh`

### Transfer to WSL (if downloaded on Windows)
From PowerShell:
```powershell
# Copy from Windows Downloads to WSL home
copy "C:\Users\YourName\Downloads\NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh" "\\wsl$\Ubuntu\home\yourusername\"
```

Or download directly in WSL:
```bash
# If you have the direct download link (requires login token)
wget -O optix_installer.sh "https://developer.download.nvidia.com/..."
```

### Install OptiX SDK
```bash
# Make executable
chmod +x NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh

# Extract to home directory
./NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh --skip-license --prefix=$HOME

# This creates: ~/NVIDIA-OptiX-SDK-7.7.0
```

### Set OptiX Environment Variable
Add to `~/.bashrc`:
```bash
export OPTIX_ROOT=~/NVIDIA-OptiX-SDK-7.7.0
export OptiX_INSTALL_DIR=$OPTIX_ROOT
```

Apply:
```bash
source ~/.bashrc
```

### Verify OptiX Installation
```bash
ls $OPTIX_ROOT/include
# Should see: optix.h, optix_types.h, etc.

ls $OPTIX_ROOT/SDK
# Should see sample projects
```

## Step 3: Install Build Tools

### CMake (3.20 or higher)
```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build
cmake --version  # Should be 3.20+
```

If CMake is too old:
```bash
# Install from Kitware repository
sudo apt-get install -y software-properties-common lsb-release
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo apt-get install -y cmake
```

### GCC/G++ Compiler
```bash
sudo apt-get install -y build-essential
g++ --version  # Should be 9.0 or higher
```

### Additional Libraries
```bash
# OpenGL Math Library
sudo apt-get install -y libglm-dev

# X11 libraries (for any GUI debug windows)
sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# Git (for cloning libraries)
sudo apt-get install -y git
```

## Step 4: Test OptiX Installation

### Build OptiX Hello World Sample
```bash
cd $OPTIX_ROOT/SDK/optixHello
mkdir build && cd build
cmake ..
make -j

# Run the sample
./optixHello

# Expected: Should output "Hello from OptiX!" and create an image
```

**If this works**: OptiX is correctly installed! ✅

## Step 5: Install Image I/O Library (STB)

### Download STB Headers (Header-only)
```bash
cd ~/
mkdir -p libs/stb
cd libs/stb
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

Add to `~/.bashrc`:
```bash
export STB_INCLUDE_DIR=~/libs/stb
```

## Common Issues & Solutions

### Issue 1: `nvidia-smi` not found
**Solution**: Install NVIDIA CUDA drivers on Windows host (not in WSL)
- Download from: https://developer.nvidia.com/cuda/wsl
- Reboot Windows after installation

### Issue 2: CUDA version mismatch
**Solution**: Ensure WSL CUDA toolkit matches host driver version
```bash
nvidia-smi  # Check "CUDA Version" in top-right
# Install matching CUDA toolkit version
```

### Issue 3: OptiX header not found during compilation
**Solution**: Verify `OPTIX_ROOT` is set correctly
```bash
echo $OPTIX_ROOT
ls $OPTIX_ROOT/include/optix.h
```

### Issue 4: PTX compilation fails
**Solution**: Check CUDA nvcc is in PATH
```bash
which nvcc
# Should show: /usr/local/cuda/bin/nvcc
```

### Issue 5: "libcudart.so.12 not found"
**Solution**: Add CUDA to library path
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

## Verification Checklist

After installation, verify everything:

```bash
# 1. CUDA
nvcc --version
nvidia-smi

# 2. OptiX
echo $OPTIX_ROOT
ls $OPTIX_ROOT/include/optix.h

# 3. Build tools
cmake --version
g++ --version

# 4. Libraries
ls /usr/include/glm
ls $STB_INCLUDE_DIR

# All should return valid output ✅
```

## Next Steps

Once verified, proceed to:
1. Create `CMakeLists.txt` for the radiosity project
2. Set up project structure (see coding_plan.md)
3. Start implementing math foundations

## Quick Reference - Environment Variables

Add all these to `~/.bashrc`:
```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# OptiX
export OPTIX_ROOT=~/NVIDIA-OptiX-SDK-7.7.0
export OptiX_INSTALL_DIR=$OPTIX_ROOT

# STB (if using)
export STB_INCLUDE_DIR=~/libs/stb
```

Apply all at once:
```bash
source ~/.bashrc
```

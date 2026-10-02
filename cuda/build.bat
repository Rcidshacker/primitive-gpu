@echo off
rem Builds gpu_search.exe. Needs CUDA Toolkit + MSVC. Change sm_89 to your GPU (RTX 40xx = sm_89, 30xx = sm_86, 20xx = sm_75).
where cl >nul 2>&1 || call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
nvcc -arch=sm_89 -O2 --fmad=false -lineinfo gpu_search.cu -o gpu_search.exe

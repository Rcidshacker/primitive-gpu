# primitive-gpu

Reproduce images with geometric primitives (triangles, circles, beziers, ...), with the shape search running on an NVIDIA GPU.

A CUDA port of the search loop from [fogleman/primitive](https://github.com/fogleman/primitive) (MIT). The greedy algorithm is the same: per step, try random shapes, hill-climb the best, commit the winner. The scoring and hill-climbing run in one persistent CUDA kernel; Go replays the chosen shapes with anti-aliasing and writes PNG / JPG / SVG / GIF.

## Build

Needs Go, the CUDA Toolkit and MSVC (Windows). Edit `sm_89` in `cuda/build.bat` for your GPU.

```
cuda\build.bat
go build -o primitive.exe .
```

## Use

```
primitive.exe -gpu cuda/gpu_search.exe -i in.jpg -o out.png -n 500 -m 1 -r 768 -s 1599
```

Flags are the original `primitive` flags plus `-gpu <path to gpu_search.exe>` and `-trace <file>` (per-step timing JSONL; summarise with `tools/analyze_trace.py`).

`-m`: 0 combo, 1 triangle, 2 rectangle, 3 ellipse, 4 circle, 5 rotated rectangle, 6 bezier, 7 rotated ellipse, 8 polygon. `-r` is the search resolution, `-s` the output size (search at 768 px, render at 4K is fine).

## Numbers

Single runs on an RTX 4050 laptop with background load, so treat them as rough.

- Triangles: about 1.1 ms/step on the GPU vs 17 to 19 ms/step on the CPU at 256 px (about 16x).
- Other modes: 6x to 24x end to end (200 shapes). Beziers about 3x.
- 4K input, 20 shapes: CPU 364 s, GPU 9.4 s.

## Limits

- Polygon mode searches convex quads only.
- The GPU scores hard edges; Go renders anti-aliased, so scores shift slightly for modes 5, 7, 8.
- A shape cannot be taller than 512 rows (taller candidates are rejected).
- Search is not bit-for-bit reproducible between runs (atomics, async hill-climb).

## License

MIT. Original work by Michael Fogleman; CUDA port and additions by Rcidshacker.

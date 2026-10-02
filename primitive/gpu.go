package primitive

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

// GPUShape is one shape proposed by cuda/gpu_search.exe (layout matches its Rec struct).
type GPUShape struct {
	Type, Alpha int32
	P           [8]float32
	R, G, B     int32
	Step        int32 // which Model.Step this shape belongs to (a step with -rep adds several)
}

// GPUSupports reports which -m modes the GPU searcher implements (all of them).
func GPUSupports(mode int) bool {
	switch mode {
	case 0, 1, 2, 3, 4, 5, 6, 7, 8:
		return true
	}
	return false
}

func writeRGBA(path string, w, h int, pix []uint8) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, []int32{int32(w), int32(h)}); err != nil {
		return err
	}
	_, err = f.Write(pix)
	return err
}

// GPUSegment asks the GPU searcher for `count` steps (each 1+rep shapes), continuing from the model's current canvas.
// The result is grouped by step.
// Search runs on hard-edged shapes; the caller replays them with AddGPU so colours and anti-aliasing are Go's.
func (model *Model) GPUSegment(exe string, count, mode, alpha, repeat int, seed int64) ([][]GPUShape, error) {
	dir, err := os.MkdirTemp("", "primitive-gpu")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	b := model.Target.Bounds()
	w, h := b.Dx(), b.Dy()
	tp, cp, op := filepath.Join(dir, "target.bin"), filepath.Join(dir, "canvas.bin"), filepath.Join(dir, "out.bin")
	if err := writeRGBA(tp, w, h, model.Target.Pix); err != nil {
		return nil, err
	}
	if err := writeRGBA(cp, w, h, model.Current.Pix); err != nil {
		return nil, err
	}
	cmd := exec.Command(exe, tp, cp, op, strconv.Itoa(count), strconv.Itoa(mode), strconv.Itoa(alpha), strconv.FormatInt(seed&0x7fffffff, 10), strconv.Itoa(repeat))
	out, err := cmd.CombinedOutput()
	Log(1, "%s", out)
	if err != nil {
		return nil, fmt.Errorf("gpu search failed: %v\n%s", err, out)
	}
	f, err := os.Open(op)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var n int32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	shapes := make([]GPUShape, n)
	if err := binary.Read(f, binary.LittleEndian, shapes); err != nil {
		return nil, err
	}
	groups := make([][]GPUShape, count)
	for _, g := range shapes {
		groups[g.Step] = append(groups[g.Step], g)
	}
	return groups, nil
}

// AddGPU turns a GPU proposal into the matching Go shape and adds it (Go recomputes the colour on its canvas).
func (model *Model) AddGPU(g GPUShape) {
	w := model.Workers[0]
	p := func(i int) int { return int(g.P[i]) }
	f := func(i int) float64 { return float64(g.P[i]) }
	var s Shape
	switch g.Type {
	case 1:
		s = &Triangle{w, p(0), p(1), p(2), p(3), p(4), p(5)}
	case 2:
		s = &Rectangle{w, p(0), p(1), p(2), p(3)}
	case 3:
		s = &Ellipse{w, p(0), p(1), p(2), p(3), false}
	case 4:
		s = &Ellipse{w, p(0), p(1), p(2), p(3), true}
	case 5:
		s = &RotatedRectangle{w, p(0), p(1), p(2), p(3), p(4)}
	case 6:
		s = &Quadratic{w, f(0), f(1), f(2), f(3), f(4), f(5), 0.5}
	case 7:
		s = &RotatedEllipse{w, f(0), f(1), f(2), f(3), f(4)}
	case 8:
		s = &Polygon{Worker: w, Order: 4, Convex: false,
			X: []float64{f(0), f(2), f(4), f(6)}, Y: []float64{f(1), f(3), f(5), f(7)}}
	default:
		panic(fmt.Sprintf("unknown gpu shape type %d", g.Type))
	}
	model.Add(s, int(g.Alpha))
}

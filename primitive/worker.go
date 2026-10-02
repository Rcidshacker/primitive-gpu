package primitive

import (
	"image"
	"math/rand"
	"time"

	"github.com/golang/freetype/raster"
)

type Worker struct {
	W, H       int
	Target     *image.RGBA
	Current    *image.RGBA
	Buffer     *image.RGBA
	Rasterizer *raster.Rasterizer
	Lines      []Scanline
	Heatmap    *Heatmap
	Rnd        *rand.Rand
	Score      float64
	Counter    int

	// harness: per-step instrumentation, reset in Init
	ClimbEvals []int // Energy() calls consumed by each HillClimb
	EvalLines  int64 // scanlines rasterized across all Energy() calls
	EvalPixels int64 // pixels covered across all Energy() calls
}

func NewWorker(target *image.RGBA) *Worker {
	w := target.Bounds().Size().X
	h := target.Bounds().Size().Y
	worker := Worker{}
	worker.W = w
	worker.H = h
	worker.Target = target
	worker.Buffer = image.NewRGBA(target.Bounds())
	worker.Rasterizer = raster.NewRasterizer(w, h)
	worker.Lines = make([]Scanline, 0, 4096) // TODO: based on height
	worker.Heatmap = NewHeatmap(w, h)
	worker.Rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
	return &worker
}

func (worker *Worker) Init(current *image.RGBA, score float64) {
	worker.Current = current
	worker.Score = score
	worker.Counter = 0
	worker.ClimbEvals = worker.ClimbEvals[:0]
	worker.EvalLines = 0
	worker.EvalPixels = 0
	worker.Heatmap.Clear()
}

func (worker *Worker) Energy(shape Shape, alpha int) float64 {
	worker.Counter++
	lines := shape.Rasterize()
	worker.EvalLines += int64(len(lines))
	for _, l := range lines {
		worker.EvalPixels += int64(l.X2 - l.X1 + 1)
	}
	// worker.Heatmap.Add(lines)
	color := computeColor(worker.Target, worker.Current, lines, alpha)
	copyLines(worker.Buffer, worker.Current, lines)
	drawLines(worker.Buffer, color, lines)
	return differencePartial(worker.Target, worker.Current, worker.Buffer, worker.Score, lines)
}

func (worker *Worker) BestHillClimbState(t ShapeType, a, n, age, m int) *State {
	var bestEnergy float64
	var bestState *State
	for i := 0; i < m; i++ {
		state := worker.BestRandomState(t, a, n)
		before := state.Energy()
		c0 := worker.Counter
		state = HillClimb(state, age).(*State)
		worker.ClimbEvals = append(worker.ClimbEvals, worker.Counter-c0)
		energy := state.Energy()
		vv("%dx random: %.6f -> %dx hill climb: %.6f\n", n, before, age, energy)
		if i == 0 || energy < bestEnergy {
			bestEnergy = energy
			bestState = state
		}
	}
	return bestState
}

func (worker *Worker) BestRandomState(t ShapeType, a, n int) *State {
	var bestEnergy float64
	var bestState *State
	for i := 0; i < n; i++ {
		state := worker.RandomState(t, a)
		energy := state.Energy()
		if i == 0 || energy < bestEnergy {
			bestEnergy = energy
			bestState = state
		}
	}
	return bestState
}

func (worker *Worker) RandomState(t ShapeType, a int) *State {
	switch t {
	default:
		return worker.RandomState(ShapeType(worker.Rnd.Intn(8)+1), a)
	case ShapeTypeTriangle:
		return NewState(worker, NewRandomTriangle(worker), a)
	case ShapeTypeRectangle:
		return NewState(worker, NewRandomRectangle(worker), a)
	case ShapeTypeEllipse:
		return NewState(worker, NewRandomEllipse(worker), a)
	case ShapeTypeCircle:
		return NewState(worker, NewRandomCircle(worker), a)
	case ShapeTypeRotatedRectangle:
		return NewState(worker, NewRandomRotatedRectangle(worker), a)
	case ShapeTypeQuadratic:
		return NewState(worker, NewRandomQuadratic(worker), a)
	case ShapeTypeRotatedEllipse:
		return NewState(worker, NewRandomRotatedEllipse(worker), a)
	case ShapeTypePolygon:
		return NewState(worker, NewRandomPolygon(worker, 4, false), a)
	}
}

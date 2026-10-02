package primitive

import (
	"encoding/json"
	"os"
)

// StepTrace is one line of the measurement harness output (JSONL).
// It exists to size a GPU port: where the time goes, how long climbs are,
// and how wide the scanline spans are (table-lookup break-even).
type StepTrace struct {
	Step       int     `json:"step"`
	SearchMs   float64 `json:"search_ms"` // runWorkers: random + hill-climb, all workers
	CommitMs   float64 `json:"commit_ms"` // model.Add: raster + colour + draw + partial diff
	Evals      int     `json:"evals"`     // Energy() calls this step, all workers
	EvalLines  int64   `json:"eval_lines"`
	EvalPixels int64   `json:"eval_pixels"`
	ClimbEvals []int   `json:"climb_evals"` // evals consumed by each hill-climb (16 per step)
	Score      float64 `json:"score"`
	// committed shape
	Lines    int `json:"lines"`
	MaxSpan  int `json:"max_span"`
	Pixels   int `json:"pixels"`
	BBoxArea int `json:"bbox_area"`
}

func shapeStats(lines []Scanline) (n, maxSpan, pixels, bbox int) {
	if len(lines) == 0 {
		return
	}
	minX, maxX := lines[0].X1, lines[0].X2
	minY, maxY := lines[0].Y, lines[0].Y
	for _, l := range lines {
		w := l.X2 - l.X1 + 1
		pixels += w
		if w > maxSpan {
			maxSpan = w
		}
		minX, maxX = minInt(minX, l.X1), maxInt(maxX, l.X2)
		minY, maxY = minInt(minY, l.Y), maxInt(maxY, l.Y)
	}
	return len(lines), maxSpan, pixels, (maxX - minX + 1) * (maxY - minY + 1)
}

func SaveTrace(path string, trace []StepTrace) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	enc := json.NewEncoder(file)
	for _, t := range trace {
		if err := enc.Encode(t); err != nil {
			return err
		}
	}
	return nil
}

package primitive

import (
	"image"
	"math/rand"
	"testing"
)

// The saved-lines score must equal the original full-copy score exactly.
func TestPartialSavedMatchesFullCopy(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	target := image.NewRGBA(image.Rect(0, 0, 96, 64))
	cur := image.NewRGBA(image.Rect(0, 0, 96, 64))
	for i := range target.Pix {
		target.Pix[i] = uint8(rnd.Intn(256))
		cur.Pix[i] = uint8(rnd.Intn(256))
	}
	w := NewWorker(target)
	w.Rnd = rnd
	score := differenceFull(target, cur)
	for n := 0; n < 200; n++ {
		var s Shape
		switch n % 3 {
		case 0:
			s = NewRandomTriangle(w)
		case 1:
			s = NewRandomEllipse(w)
		default:
			s = NewRandomQuadratic(w) // anti-aliased spans
		}
		lines := s.Rasterize()
		color := computeColor(target, cur, lines, 128)
		before := copyRGBA(cur)
		saved := saveLines(cur, lines)
		drawLines(cur, color, lines)
		a := differencePartial(target, before, cur, score, lines)
		b := differencePartialSaved(target, saved, cur, score, lines)
		if a != b {
			t.Fatalf("shape %d: full-copy %.12f != saved-lines %.12f", n, a, b)
		}
		score = a
	}
}

package primitive

import (
	"image"
	"math"
)

// saveLines copies only the pixels a shape is about to change, in scanline order.
// Model.Add used to copy the whole canvas for every shape, which costs O(image) per shape (133 MB at 8K).
func saveLines(im *image.RGBA, lines []Scanline) []uint8 {
	n := 0
	for _, line := range lines {
		n += (line.X2 - line.X1 + 1) * 4
	}
	saved := make([]uint8, 0, n)
	for _, line := range lines {
		a := im.PixOffset(line.X1, line.Y)
		saved = append(saved, im.Pix[a:a+(line.X2-line.X1+1)*4]...)
	}
	return saved
}

// differencePartialSaved is differencePartial, reading the "before" pixels from saveLines output.
func differencePartialSaved(target *image.RGBA, saved []uint8, after *image.RGBA, score float64, lines []Scanline) float64 {
	size := target.Bounds().Size()
	w, h := size.X, size.Y
	total := uint64(math.Pow(score*255, 2) * float64(w*h*4))
	k := 0
	for _, line := range lines {
		i := target.PixOffset(line.X1, line.Y)
		for x := line.X1; x <= line.X2; x++ {
			tr, tg, tb, ta := int(target.Pix[i]), int(target.Pix[i+1]), int(target.Pix[i+2]), int(target.Pix[i+3])
			br, bg, bb, ba := int(saved[k]), int(saved[k+1]), int(saved[k+2]), int(saved[k+3])
			ar, ag, ab, aa := int(after.Pix[i]), int(after.Pix[i+1]), int(after.Pix[i+2]), int(after.Pix[i+3])
			i += 4
			k += 4
			dr1, dg1, db1, da1 := tr-br, tg-bg, tb-bb, ta-ba
			dr2, dg2, db2, da2 := tr-ar, tg-ag, tb-ab, ta-aa
			total -= uint64(dr1*dr1 + dg1*dg1 + db1*db1 + da1*da1)
			total += uint64(dr2*dr2 + dg2*dg2 + db2*db2 + da2*da2)
		}
	}
	return math.Sqrt(float64(total)/float64(w*h*4)) / 255
}

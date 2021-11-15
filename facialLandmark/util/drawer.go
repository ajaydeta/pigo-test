package util

import (
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
	"image/color"
	"math"
)

var (
	red    = gg.NewSolidPattern(color.RGBA{R: 255, G: 0, B: 0, A: 255})
	green  = gg.NewSolidPattern(color.RGBA{R: 0, G: 255, B: 0, A: 255})
	blue   = gg.NewSolidPattern(color.RGBA{R: 0, G: 0, B: 255, A: 255})
	yellow = gg.NewSolidPattern(color.RGBA{R: 255, G: 255, B: 0, A: 255})
)

func DrawFace(result pigo.Detection, dc *gg.Context) {
	dc.DrawArc(
		float64(result.Col),
		float64(result.Row),
		float64(result.Scale/2),
		0,
		2*math.Pi,
	)
	dc.SetLineWidth(10.0)
	dc.SetStrokeStyle(red)
	dc.Stroke()
}

func DrawEyes(result *pigo.Puploc, dc *gg.Context) {
	dc.DrawRectangle(
		float64(float32(result.Col)-result.Scale/2),
		float64(float32(result.Row)-result.Scale/2),
		float64(result.Scale),
		float64(result.Scale),
	)
	dc.SetLineWidth(1.0)
	dc.SetStrokeStyle(green)
	dc.Stroke()
}

func DrawLandmark(result *pigo.Puploc, dc *gg.Context) {
	dc.DrawPoint(
		float64(result.Col),
		float64(result.Row),
		float64(result.Scale/10),
	)
	dc.SetLineWidth(1.0)
	dc.SetFillStyle(yellow)
	dc.Fill()
}

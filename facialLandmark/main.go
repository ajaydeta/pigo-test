package main

import (
	"fmt"
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
	"image/color"
	"io/ioutil"
	"log"
	"math"
)

var (
	faceClassifier   *pigo.Pigo
	puplocClassifier *pigo.PuplocCascade
	flpcs            map[string][]*pigo.FlpCascade
	cParams          pigo.CascadeParams
	imgParams        pigo.ImageParams
	dc               *gg.Context
)

var (
	eyeCascades  = []string{"lp46", "lp44", "lp42", "lp38", "lp312"}
	mouthCascade = []string{"lp93", "lp84", "lp82", "lp81"}
)

func main() {
	src, err := pigo.GetImage("facialLandmark/image/tes.png")
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	dc = gg.NewContext(cols, rows)
	dc.DrawImage(src, 0, 0)

	results := faceDetect(pixels, rows, cols)
	err = detsSetup(results)
	if err != nil {
		log.Fatalf("setuop: %v", err)
	}

}

func faceDetect(pixels []uint8, rows, cols int) []pigo.Detection {
	imgParams = pigo.ImageParams{
		Pixels: pixels,
		Rows:   rows,
		Cols:   cols,
		Dim:    cols,
	}

	cParams = pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1100,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,
		ImageParams: imgParams,
	}

	faceCascade, err := ioutil.ReadFile("cascade/facefinder")
	if err != nil {
		log.Fatalln("load face cascade", err.Error())
		return nil
	}

	//unpack face cascade file
	p := pigo.NewPigo()
	faceClassifier, err = p.Unpack(faceCascade)
	if err != nil {
		log.Fatalln("unpack cascade face file", err.Error())
		return nil
	}

	puplocCascade, err := ioutil.ReadFile("cascade/puploc")
	if err != nil {
		log.Fatalf("Error reading the puploc cascade file: %s", err)
		return nil
	}

	puplocClassifier, err = puplocClassifier.UnpackCascade(puplocCascade)
	if err != nil {
		log.Fatalf("Error unpacking the puploc cascade file: %s", err)
		return nil
	}

	flpcs, err = puplocClassifier.ReadCascadeDir("cascade/lps")
	if err != nil {
		log.Fatalf("Error unpacking the facial landmark detection cascades: %s", err)
	}

	dets := faceClassifier.RunCascade(cParams, 0.0)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = faceClassifier.ClusterDetections(dets, 0.2)
	return dets
}

func detsSetup(results []pigo.Detection) error {
	var qThresh float32 = 7.5

	fmt.Println(results)
	for i := 0; i < len(results); i++ {
		result := results[i]
		if result.Q > qThresh {
			dc.DrawArc(
				float64(result.Col),
				float64(result.Row),
				float64(result.Scale/2),
				0,
				2*math.Pi,
			)
			dc.SetLineWidth(1.0)
			dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 255, G: 0, B: 0, A: 255}))
			dc.Stroke()

			// left eye
			puploc := &pigo.Puploc{
				Row:      result.Row - int(0.085*float32(result.Scale)),
				Col:      result.Col - int(0.185*float32(result.Scale)),
				Scale:    float32(result.Scale) * 0.4,
				Perturbs: 63,
			}
			leftEye := puplocClassifier.RunDetector(*puploc, imgParams, 0.0, false)
			if leftEye.Row > 0 && leftEye.Col > 0 {
				dc.DrawRectangle(
					float64(float32(leftEye.Col)-leftEye.Scale/2),
					float64(float32(leftEye.Row)-leftEye.Scale/2),
					float64(leftEye.Scale),
					float64(leftEye.Scale),
				)
				dc.SetLineWidth(1.0)
				dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 255, B: 0, A: 255}))
				dc.Stroke()
			}

			// right eye
			puploc = &pigo.Puploc{
				Row:      results[i].Row - int(0.085*float32(results[i].Scale)),
				Col:      results[i].Col + int(0.185*float32(results[i].Scale)),
				Scale:    float32(results[i].Scale) * 0.4,
				Perturbs: 63,
			}

			rightEye := puplocClassifier.RunDetector(*puploc, imgParams, 0.0, false)
			if rightEye.Row > 0 && rightEye.Col > 0 {
				dc.DrawRectangle(
					float64(float32(rightEye.Col)-rightEye.Scale/2),
					float64(float32(rightEye.Row)-rightEye.Scale/2),
					float64(rightEye.Scale),
					float64(rightEye.Scale),
				)
				dc.SetLineWidth(1.0)
				dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 255, B: 0, A: 255}))
				dc.Stroke()
			}

			for _, eye := range eyeCascades {
				for _, flpc := range flpcs[eye] {
					flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
					if flp.Row > 0 && flp.Col > 0 {
						dc.DrawArc(
							float64(flp.Col),
							float64(flp.Row),
							float64(flp.Scale/2),
							0,
							2*math.Pi,
						)
						dc.SetLineWidth(1.0)
						dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 0, B: 255, A: 255}))
						dc.Stroke()
					}

					flp = flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
					if flp.Row > 0 && flp.Col > 0 {
						dc.DrawArc(
							float64(flp.Col),
							float64(flp.Row),
							float64(flp.Scale/2),
							0,
							2*math.Pi,
						)
						dc.SetLineWidth(1.0)
						dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 0, B: 255, A: 255}))
						dc.Stroke()
					}
				}
			}

			for _, mouth := range mouthCascade {
				for _, flpc := range flpcs[mouth] {
					flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
					if flp.Row > 0 && flp.Col > 0 {
						dc.DrawArc(
							float64(flp.Col),
							float64(flp.Row),
							float64(flp.Scale/2),
							0,
							2*math.Pi,
						)
						dc.SetLineWidth(1.0)
						dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 0, B: 255, A: 255}))
						dc.Stroke()
					}
				}
			}
			flp := flpcs["lp84"][0].GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
			if flp.Row > 0 && flp.Col > 0 {
				dc.DrawArc(
					float64(flp.Col),
					float64(flp.Row),
					float64(flp.Scale/2),
					0,
					2*math.Pi,
				)
				dc.SetLineWidth(1.0)
				dc.SetStrokeStyle(gg.NewSolidPattern(color.RGBA{R: 0, G: 0, B: 255, A: 255}))
				dc.Stroke()
			}
		}
	}
	err := dc.SavePNG("facialLandmark/image/jos.png")
	if err != nil {
		return err
	}
	return nil
}

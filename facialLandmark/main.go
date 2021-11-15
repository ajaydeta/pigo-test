package main

import (
	"fmt"
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
	"io/ioutil"
	"log"
	"pigo-test/facialLandmark/util"
)

var (
	faceClassifier   *pigo.Pigo
	puplocClassifier *pigo.PuplocCascade
	flpcs            map[string][]*pigo.FlpCascade
	cParams          pigo.CascadeParams
	imgParams        pigo.ImageParams
	dc               *gg.Context
)

const (
	rowTimes   = 0.085
	colTimes   = 0.185
	scaleTimes = 0.4
	perturbs   = 63
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
	err = detectionDrawer(results)
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

func detectionDrawer(results []pigo.Detection) error {
	var qThresh float32 = 7.5

	fmt.Println(results)
	for i := 0; i < len(results); i++ {
		result := results[i]
		if result.Q > qThresh {
			util.DrawFace(result, dc)

			leftEye, rightEye, puploc := pupLoc(result)

			for _, eye := range eyeCascades {
				for _, flpc := range flpcs[eye] {
					flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
					if flp.Row > 0 && flp.Col > 0 {
						util.DrawLandmark(flp, dc)
					}

					flp = flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
					if flp.Row > 0 && flp.Col > 0 {
						util.DrawLandmark(flp, dc)
					}
				}
			}

			for _, mouth := range mouthCascade {
				for _, flpc := range flpcs[mouth] {
					flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
					if flp.Row > 0 && flp.Col > 0 {
						util.DrawLandmark(flp, dc)
					}
				}
			}
			flp := flpcs["lp84"][0].GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
			if flp.Row > 0 && flp.Col > 0 {
				util.DrawLandmark(flp, dc)
			}
		}
	}
	err := dc.SavePNG("facialLandmark/image/jos.png")
	if err != nil {
		return err
	}
	return nil
}

func pupLoc(result pigo.Detection) (leftEye *pigo.Puploc, rightEye *pigo.Puploc, puploc *pigo.Puploc) {
	// left eye
	puploc = &pigo.Puploc{
		Row:      result.Row - int(rowTimes*float32(result.Scale)),
		Col:      result.Col - int(colTimes*float32(result.Scale)),
		Scale:    float32(result.Scale) * scaleTimes,
		Perturbs: perturbs,
	}
	leftEye = puplocClassifier.RunDetector(*puploc, imgParams, 0.0, false)
	if leftEye.Row > 0 && leftEye.Col > 0 {
		util.DrawEyes(leftEye, dc)
	}

	// right eye
	puploc = &pigo.Puploc{
		Row:      result.Row - int(rowTimes*float32(result.Scale)),
		Col:      result.Col + int(colTimes*float32(result.Scale)),
		Scale:    float32(result.Scale) * scaleTimes,
		Perturbs: perturbs,
	}

	rightEye = puplocClassifier.RunDetector(*puploc, imgParams, 0.0, false)
	if rightEye.Row > 0 && rightEye.Col > 0 {
		util.DrawEyes(rightEye, dc)
	}
	return leftEye, rightEye, puploc
}

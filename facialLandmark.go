package main

import (
	pigo "github.com/esimov/pigo/core"
	"github.com/fogleman/gg"
	"io/ioutil"
	"log"
)

var (
	faceClassifier   *pigo.Pigo
	puplocClassifier *pigo.PuplocCascade
	flpcs            map[string][]*pigo.FlpCascade
	cParams          pigo.CascadeParams
	imgParams        pigo.ImageParams
)

var (
	eyeCascades  = []string{"lp46", "lp44", "lp42", "lp38", "lp312"}
	mouthCascade = []string{"lp93", "lp84", "lp82", "lp81"}
)

func main() {
	src, err := pigo.GetImage("./image/tes.png")
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	dc = gg.NewContext(cols, rows)
	dc.DrawImage(src, 0, 0)

	results := faceDetect(pixels, rows, cols)

	dets := make([][]int, len(results))

	for i := 0; i < len(results); i++ {
		dets[i] = append(dets[i], results[i].Row, results[i].Col, results[i].Scale, int(results[i].Q), 0)
		// left eye
		puploc := &pigo.Puploc{
			Row:      results[i].Row - int(0.085*float32(results[i].Scale)),
			Col:      results[i].Col - int(0.185*float32(results[i].Scale)),
			Scale:    float32(results[i].Scale) * 0.4,
			Perturbs: 63,
		}
		leftEye := puplocClassifier.RunDetector(*puploc, imgParams, 0.0, false)
		if leftEye.Row > 0 && leftEye.Col > 0 {
			dets[i] = append(dets[i], leftEye.Row, leftEye.Col, int(leftEye.Scale), int(results[i].Q), 1)
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
			dets[i] = append(dets[i], rightEye.Row, rightEye.Col, int(rightEye.Scale), int(results[i].Q), 1)
		}

		for _, eye := range eyeCascades {
			for _, flpc := range flpcs[eye] {
				flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
				if flp.Row > 0 && flp.Col > 0 {
					dets[i] = append(dets[i], flp.Row, flp.Col, int(flp.Scale), int(results[i].Q), 2)
				}

				flp = flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
				if flp.Row > 0 && flp.Col > 0 {
					dets[i] = append(dets[i], flp.Row, flp.Col, int(flp.Scale), int(results[i].Q), 2)
				}
			}
		}

		// Traverse all the mouth cascades and run the detector on each of them.
		for _, mouth := range mouthCascade {
			for _, flpc := range flpcs[mouth] {
				flp := flpc.GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, false)
				if flp.Row > 0 && flp.Col > 0 {
					dets[i] = append(dets[i], flp.Row, flp.Col, int(flp.Scale), int(results[i].Q), 2)
				}
			}
		}
		flp := flpcs["lp84"][0].GetLandmarkPoint(leftEye, rightEye, imgParams, puploc.Perturbs, true)
		if flp.Row > 0 && flp.Col > 0 {
			dets[i] = append(dets[i], flp.Row, flp.Col, int(flp.Scale), int(results[i].Q), 2)
		}
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
		MinSize:     60,
		MaxSize:     600,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,
		ImageParams: imgParams,
	}

	faceCascade, err := ioutil.ReadFile("./cascade/facefinder")
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

	puplocCascade, err := ioutil.ReadFile("./cascade/puploc")
	if err != nil {
		log.Fatalf("Error reading the puploc cascade file: %s", err)
		return nil
	}

	puplocClassifier, err = puplocClassifier.UnpackCascade(puplocCascade)
	if err != nil {
		log.Fatalf("Error unpacking the puploc cascade file: %s", err)
		return nil
	}

	flpcs, err = puplocClassifier.ReadCascadeDir("../../cascade/lps")
	if err != nil {
		log.Fatalf("Error unpacking the facial landmark detection cascades: %s", err)
	}

	dets := faceClassifier.RunCascade(cParams, 0.0)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = faceClassifier.ClusterDetections(dets, 0.0)
	return dets
}

package adapter

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLinearRegressionUsingMat(t *testing.T) {
	rand.Seed(123)
	N := 1000
	lambda := 1.0

	theta := mat.NewDense(3, 1, []float64{2.0, 3.0, 4.0})
	X, Y := syntheticData(N, theta)

	XTX := mat.NewDense(3, 3, nil)
	XTX.Product(X.T(), X)

	I := mat.NewDiagDense(3, []float64{1.0, 1.0, 1.0})
	reg := mat.DenseCopyOf(I)
	reg.Scale(lambda, reg)

	XTX.Add(XTX, reg)
	XTX.Inverse(XTX)

	XTY := mat.NewDense(3, 1, nil)
	XTY.Product(X.T(), Y)

	estimated := mat.NewDense(3, 1, nil)
	estimated.Product(XTX, XTY)

	if !mat.EqualApprox(theta, estimated, 0.05) {
		t.Errorf("Parameter should be estimated.")
	}
}

func TestLinearRegressionUsingAdaptedMat(t *testing.T) {
	rand.Seed(123)
	N := 1000
	lambda := 1.0

	theta := mat.NewDense(3, 1, []float64{2.0, 3.0, 4.0})
	X, Y := syntheticData(N, theta)

	I := mat.NewDiagDense(3, []float64{1.0, 1.0, 1.0})
	reg := DenseCopyOf(I).Scale(lambda)

	XTXinv, _ := X.Transpose().Product(X).Add(reg).Inverse()
	XTY := X.Transpose().Product(Y)

	estimated := XTXinv.Product(XTY)

	if !mat.EqualApprox(theta, estimated, 0.05) {
		t.Errorf("Parameter should be estimated.")
	}
}

func uniform(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func syntheticData(N int, theta *mat.Dense) (*Dense, *Dense) {
	X := mat.NewDense(N, 3, nil)
	for i := 0; i < N; i++ {
		X.SetRow(i, []float64{1.0, uniform(-10.0, 10.0), uniform(-10.0, 10.0)})
	}

	Y := mat.NewDense(N, 1, nil)
	Y.Product(X, theta)

	epsilon := mat.NewDense(N, 1, nil)
	for i := 0; i < N; i++ {
		epsilon.Set(i, 0, rand.NormFloat64())
	}
	Y.Add(Y, epsilon)

	return &Dense{X}, &Dense{Y}
}

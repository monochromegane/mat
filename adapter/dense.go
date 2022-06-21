package adapter

import "gonum.org/v1/gonum/mat"

type Dense struct {
	*mat.Dense
}

func DenseCopyOf(a mat.Matrix) *Dense {
	return &Dense{mat.DenseCopyOf(a)}
}

func NewDense(r, c int, data []float64) *Dense {
	return &Dense{mat.NewDense(r, c, data)}
}

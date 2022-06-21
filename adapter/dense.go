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

func (m *Dense) Add(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Add(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) Apply(fn func(i, j int, v float64) float64) mat.Matrix {
	var dense mat.Dense
	dense.Apply(fn, m.Dense)
	return &Dense{&dense}
}

func (m *Dense) Augment(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Augment(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) CloneFrom() mat.Matrix {
	var dense mat.Dense
	dense.CloneFrom(m.Dense)
	return &Dense{&dense}
}

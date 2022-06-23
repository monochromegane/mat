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

// func (m *Dense) ColView(j int) mat.Vector

// func (m *Dense) DiagView() mat.Diagonal

func (m *Dense) DivElem(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.DivElem(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) Exp() mat.Matrix {
	var dense mat.Dense
	dense.Exp(m.Dense)
	return &Dense{&dense}
}

func (m *Dense) Grow(r, c int) mat.Matrix {
	dense := m.Dense.Grow(r, c).(*mat.Dense)
	return &Dense{dense}
}

func (m *Dense) Inverse() (mat.Matrix, error) {
	var dense mat.Dense
	err := dense.Inverse(m.Dense)
	return &Dense{&dense}, err
}

func (m *Dense) Kronecker(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Kronecker(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) Mul(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Mul(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) MulElem(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.MulElem(m.Dense, b)
	return &Dense{&dense}
}

// func (m *Dense) Outer(alpha float64, x, y Vector)

func (m *Dense) Pow(n int) mat.Matrix {
	var dense mat.Dense
	dense.Pow(m.Dense, n)
	return &Dense{&dense}
}

func (m *Dense) Product(factors ...mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Product(append([]mat.Matrix{m.Dense}, factors...)...)
	return &Dense{&dense}
}

func (m *Dense) RankOne(alpha float64, x, y mat.Vector) mat.Matrix {
	var dense mat.Dense
	dense.RankOne(m.Dense, alpha, x, y)
	return &Dense{&dense}
}

// func (m *Dense) RowView(i int) Vector

func (m *Dense) Scale(f float64) mat.Matrix {
	var dense mat.Dense
	dense.Scale(f, m.Dense)
	return &Dense{&dense}
}

func (m *Dense) Slice(i, k, j, l int) mat.Matrix {
	dense := m.Dense.Slice(i, k, j, l).(*mat.Dense)
	return &Dense{dense}
}

func (m *Dense) Solve(b mat.Matrix) (mat.Matrix, error) {
	var dense mat.Dense
	err := dense.Solve(m.Dense, b)
	return &Dense{&dense}, err
}

func (m *Dense) Stack(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Stack(m.Dense, b)
	return &Dense{&dense}
}

func (m *Dense) Sub(b mat.Matrix) mat.Matrix {
	var dense mat.Dense
	dense.Sub(m.Dense, b)
	return &Dense{&dense}
}

// func (m *Dense) T() mat.Matrix

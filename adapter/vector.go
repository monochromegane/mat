package adapter

import "gonum.org/v1/gonum/mat"

type VecDense struct {
	*mat.VecDense
}

func NewVecDense(n int, data []float64) *VecDense {
	return &VecDense{mat.NewVecDense(n, data)}
}

func VecDenseCopyOf(a mat.Vector) *VecDense {
	return &VecDense{mat.VecDenseCopyOf(a)}
}

func (v *VecDense) AddScaledVec(alpha float64, b mat.Vector) mat.Vector {
	var vec mat.VecDense
	vec.AddScaledVec(v.VecDense, alpha, b)
	return &VecDense{&vec}
}

func (v *VecDense) AddVec(b mat.Vector) mat.Vector {
	var vec mat.VecDense
	vec.AddVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) CloneFromVec() mat.Vector {
	var vec mat.VecDense
	vec.CloneFromVec(v.VecDense)
	return &VecDense{&vec}
}

func (v *VecDense) DivElemVec(b mat.Vector) mat.Vector {
	var vec mat.VecDense
	vec.DivElemVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) MulElemVec(b mat.Vector) mat.Vector {
	var vec mat.VecDense
	vec.MulElemVec(v.VecDense, b)
	return &VecDense{&vec}
}

// func (v *VecDense) MulVec(a Matrix, b Vector)

func (v *VecDense) ScaleVec(alpha float64) mat.Vector {
	var vec mat.VecDense
	vec.ScaleVec(alpha, v.VecDense)
	return &VecDense{&vec}
}

func (v *VecDense) SliceVec(i, k int) mat.Vector {
	vec := v.VecDense.SliceVec(i, k).(*mat.VecDense)
	return &VecDense{vec}
}

// func (v *VecDense) SolveVec(a Matrix, b Vector) error

func (v *VecDense) SubVec(b mat.Vector) mat.Vector {
	var vec mat.VecDense
	vec.SubVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) T() mat.Matrix {
	return mat.Transpose{v}
}

func (v *VecDense) TVec() mat.Vector {
	return mat.TransposeVec{v}
}

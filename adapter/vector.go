package adapter

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type VecDense struct {
	*mat.VecDense
}

func NewVecDense(n int, data []float64) *VecDense {
	return &VecDense{mat.NewVecDense(n, data)}
}

func VecDenseCopyOf(a mat.Vector) *VecDense {
	return &VecDense{mat.VecDenseCopyOf(a)}
}

func (v *VecDense) AddScaledVec(alpha float64, b mat.Vector) *VecDense {
	var vec mat.VecDense
	vec.AddScaledVec(v.VecDense, alpha, b)
	return &VecDense{&vec}
}

func (v *VecDense) AddVec(b mat.Vector) *VecDense {
	var vec mat.VecDense
	vec.AddVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) CloneFromVec() *VecDense {
	var vec mat.VecDense
	vec.CloneFromVec(v.VecDense)
	return &VecDense{&vec}
}

func (v *VecDense) DivElemVec(b mat.Vector) *VecDense {
	var vec mat.VecDense
	vec.DivElemVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) MulElemVec(b mat.Vector) *VecDense {
	var vec mat.VecDense
	vec.MulElemVec(v.VecDense, b)
	return &VecDense{&vec}
}

func (v *VecDense) MulVec(b mat.Vector) *VecDense {
	var vec mat.VecDense
	vec.MulVec(v.VecDense.TVec(), b)
	return &VecDense{&vec}
}

func (v *VecDense) ScaleVec(alpha float64) *VecDense {
	var vec mat.VecDense
	vec.ScaleVec(alpha, v.VecDense)
	return &VecDense{&vec}
}

func (v *VecDense) SliceVec(i, k int) *VecDense {
	vec := v.VecDense.SliceVec(i, k).(*mat.VecDense)
	return &VecDense{vec}
}

// func (v *VecDense) SolveVec(a Matrix, b Vector) error

func (v *VecDense) SubVec(b mat.Vector) *VecDense {
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

func (v *VecDense) Transpose() *Dense {
	return &Dense{mat.DenseCopyOf(v.VecDense)}
}

func (v *VecDense) String() string {
	return fmt.Sprintf("%v", mat.Formatted(v.VecDense, mat.Prefix(""), mat.Squeeze()))
}

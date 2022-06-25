package adapter

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVecDense(t *testing.T) {
	vec := &VecDense{mat.NewVecDense(4, data)}
	if _, ok := interface{}(vec).(mat.Vector); !ok {
		t.Errorf("VecDense should implement mat.Vector")
	}
}

func TestNewVecDense(t *testing.T) {
	org := mat.NewVecDense(4, data)
	adapted := NewVecDense(4, data)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of NewVecDense should be equal.")
	}
}

func TestVecDenseCopyOf(t *testing.T) {
	a := mat.NewVecDense(4, data)

	org := mat.VecDenseCopyOf(a)
	adapted := VecDenseCopyOf(a)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of VecDenseCopyOf should be equal.")
	}
}

func TestVecDenseAddScaledVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.AddScaledVec(a, 1.1, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.AddScaledVec(1.1, b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of AddScaledVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.AddScaledVec(1.1, bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of AddScaledVec should be equal.")
	}
}

func TestVecDenseAddVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.AddVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.AddVec(b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of AddVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.AddVec(bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of AddVec should be equal.")
	}
}

func TestVecDenseCloneFromVec(t *testing.T) {
	a := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.CloneFromVec(a)

	ax := VecDenseCopyOf(a)
	adapted := ax.CloneFromVec()

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of CloneFromVec should be equal.")
	}
}

func TestVecDenseDivElemVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.DivElemVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.DivElemVec(b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of DivElemVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.DivElemVec(bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of DivElemVec should be equal.")
	}
}

func TestVecDenseMulElemVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.MulElemVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.MulElemVec(b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of MulElemVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.MulElemVec(bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of MulElemVec should be equal.")
	}
}

func TestVecDenseMulVec(t *testing.T) {
	a := mat.NewVecDense(4, data).TVec()
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(1, nil)
	org.MulVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.MulVec(b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of MulVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.MulVec(bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of MulVec should be equal.")
	}
}

func TestVecDenseScale(t *testing.T) {
	a := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.ScaleVec(2.0, a)

	ax := VecDenseCopyOf(a)
	adapted := ax.ScaleVec(2.0)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of ScaleVec should be equal.")
	}
}

func TestDenseSliceVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	org := a.SliceVec(1, 2)

	ax := VecDenseCopyOf(a)
	adapted := ax.SliceVec(1, 2)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of SliceVec should be equal.")
	}
}

func TestVecDenseSolveVec(t *testing.T) {
	a := mat.NewVecDense(1, []float64{2.0})
	b := mat.NewVecDense(1, []float64{1.0})

	org := mat.NewVecDense(1, nil)
	errOrg := org.SolveVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted, errAdapted := ax.SolveVec(b)

	if errOrg != nil || errAdapted != nil {
		t.Errorf("SolveVec should not return err.")
	}

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of SolveVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted, errAdapted = ax.SolveVec(bx)

	if errAdapted != nil {
		t.Errorf("SolveVec should not return err.")
	}

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of MulVec should be equal.")
	}
}

func TestVecDenseSubVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	b := mat.NewVecDense(4, data)

	org := mat.NewVecDense(4, nil)
	org.SubVec(a, b)

	ax := VecDenseCopyOf(a)
	adapted := ax.SubVec(b)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of SubVec should be equal.")
	}

	ax = VecDenseCopyOf(a)
	bx := VecDenseCopyOf(b)
	adapted = ax.SubVec(bx)

	if !mat.EqualApprox(org, adapted.VecDense, epsilon) {
		t.Errorf("Result of SubVec should be equal.")
	}
}

func TestVecDenseT(t *testing.T) {
	a := mat.NewVecDense(4, data)
	org := a.T()

	ax := VecDenseCopyOf(a)
	adapted := ax.T()

	if !mat.EqualApprox(org, adapted, epsilon) {
		t.Errorf("Result of T should be equal.")
	}
}

func TestVecDenseTVec(t *testing.T) {
	a := mat.NewVecDense(4, data)
	org := a.TVec()

	ax := VecDenseCopyOf(a)
	adapted := ax.TVec()

	if !mat.EqualApprox(org, adapted, epsilon) {
		t.Errorf("Result of TVec should be equal.")
	}
}

func TestVecDenseTranspose(t *testing.T) {
	a := mat.NewVecDense(4, data)
	org := mat.DenseCopyOf(a)

	ax := VecDenseCopyOf(a)
	adapted := ax.Transpose()

	if !mat.EqualApprox(org, adapted.Dense, epsilon) {
		t.Errorf("Result of Transpose should be equal.")
	}
}

func ExampleVecDenseString() {
	ax := NewVecDense(4, data)
	fmt.Printf("%v", ax)
	// Output:
	// ⎡1⎤
	// ⎢2⎥
	// ⎢3⎥
	// ⎣4⎦
}

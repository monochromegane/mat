package adapter

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

var data = []float64{1.0, 2.0, 3.0, 4.0}
var epsilon = 1e-14

func TestDense(t *testing.T) {
	dense := &Dense{mat.NewDense(2, 2, data)}
	if _, ok := interface{}(dense).(mat.Matrix); !ok {
		t.Errorf("Dense should implement mat.Matrix")
	}
}

func TestDenseCopyOf(t *testing.T) {
	a := mat.NewDense(2, 2, data)

	org := mat.DenseCopyOf(a)
	adapted := DenseCopyOf(a)

	if !mat.EqualApprox(org, adapted.Dense, epsilon) {
		t.Errorf("Result of DenseCopyOf should be equal.")
	}
}

func TestNewDense(t *testing.T) {
	org := mat.NewDense(2, 2, data)
	adapted := NewDense(2, 2, data)

	if !mat.EqualApprox(org, adapted.Dense, epsilon) {
		t.Errorf("Result of NewDense should be equal.")
	}
}

func TestDenseAdd(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.Add(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.Add(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Add should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.Add(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Add should be equal.")
	}
}

func TestDenseApply(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	fn := func(i, j int, v float64) float64 { return v * 2.0 }

	org := mat.NewDense(2, 2, nil)
	org.Apply(fn, a)

	ax := DenseCopyOf(a)
	adapted := ax.Apply(fn)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Apply should be equal.")
	}
}

func TestDenseAugment(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 4, nil)
	org.Augment(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.Augment(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Augment should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.Augment(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Augment should be equal.")
	}
}

func TestDenseCloneFrom(t *testing.T) {
	a := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.CloneFrom(a)

	ax := DenseCopyOf(a)
	adapted := ax.CloneFrom()

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of CloneFrom should be equal.")
	}
}

func TestDenseDivElem(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.DivElem(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.DivElem(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of DivElem should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.DivElem(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of DivElem should be equal.")
	}
}

func TestDenseExp(t *testing.T) {
	a := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.Exp(a)

	ax := DenseCopyOf(a)
	adapted := ax.Exp()

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Exp should be equal.")
	}
}

func TestDenseGrow(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	org := a.Grow(3, 3)

	ax := NewDense(2, 2, data)
	adapted := ax.Grow(3, 3)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Grow should be equal.")
	}
}

func TestDenseInverse(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	org := mat.NewDense(2, 2, nil)
	errOrg := org.Inverse(a)

	ax := NewDense(2, 2, data)
	adapted, errAdapted := ax.Inverse()

	if errOrg != nil || errAdapted != nil {
		t.Errorf("Inverse should not return err.")
	}

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Inverse should be equal.")
	}
}

func TestDenseKronecker(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(4, 4, nil)
	org.Kronecker(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.Kronecker(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Kronecker should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.Kronecker(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Kronecker should be equal.")
	}
}

func TestDenseMul(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.Mul(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.Mul(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Mul should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.Mul(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Mul should be equal.")
	}
}

func TestDenseMulElem(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.MulElem(a, b)

	ax := DenseCopyOf(a)
	adapted := ax.MulElem(b)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of MulElem should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	adapted = ax.MulElem(bx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of MulElem should be equal.")
	}
}

func TestDensePow(t *testing.T) {
	a := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.Pow(a, 2)

	ax := DenseCopyOf(a)
	adapted := ax.Pow(2)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Pow should be equal.")
	}
}

func TestDenseProduct(t *testing.T) {
	a := mat.NewDense(2, 2, data)
	b := mat.NewDense(2, 2, data)
	c := mat.NewDense(2, 2, data)

	org := mat.NewDense(2, 2, nil)
	org.Product(a, b, c)

	ax := DenseCopyOf(a)
	adapted := ax.Product(b, c)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Product should be equal.")
	}

	ax = DenseCopyOf(a)
	bx := DenseCopyOf(b)
	cx := DenseCopyOf(c)
	adapted = ax.Product(bx, cx)

	if !mat.EqualApprox(org, adapted.(*Dense).Dense, epsilon) {
		t.Errorf("Result of Product should be equal.")
	}
}

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

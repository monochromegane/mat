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

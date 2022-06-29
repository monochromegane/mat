# Matrix adapter

The small adapter which provides method signatures that allow intuitive operation with fewer lines of code for [gonum/mat](https://github.com/gonum/gonum/tree/master/mat).

## Example

Comparison of implementaion for normal equation of Ridge regression ($\hat{\theta} = (X^{\top}X + \lambda I)^{-1} X^{\top}Y$).

### Matrix adapter

```go
	X, Y := syntheticData(N, theta)

	I := mat.NewDiagDense(3, []float64{1.0, 1.0, 1.0})
	reg := DenseCopyOf(I).Scale(lambda)

	XTXinv, _ := X.Transpose().Product(X).Add(reg).Inverse()
	XTY := X.Transpose().Product(Y)

	estimated := XTXinv.Product(XTY)
```

### gonum/mat

```go
	X, Y := syntheticData(N, theta)

	XTXinv := mat.NewDense(3, 3, nil)
	XTXinv.Product(X.T(), X)

	I := mat.NewDiagDense(3, []float64{1.0, 1.0, 1.0})
	reg := mat.DenseCopyOf(I)
	reg.Scale(lambda, reg)

	XTXinv.Add(XTXinv, reg)
	XTXinv.Inverse(XTXinv)

	XTY := mat.NewDense(3, 1, nil)
	XTY.Product(X.T(), Y)

	estimated := mat.NewDense(3, 1, nil)
	estimated.Product(XTXinv, XTY)
```

## Characteristics of the adapter

### New input argment

The receiver itself is not included in the input argument.
For example, ever `c.Product(a, b)` can be written as `c := a.Product(b)`.
Of course, all matrices, including the receiver, are invariant to operations.

### New return value

The new function now returns a struct as the result of the operation.
This struct is wrapped as the adapter, but implements the mat.Matrix interface.

Both the input argument and return value of the new function follow the mat.Matrix (or mat.Vector) interface, making it easy to use with existing Matrix.

### fmt.Stringer interface

String() outputs the contents of a well-formed matrix as follw:

```go
a := NewDense(2, 2, data)
fmt.Printf("%v", a)
// ⎡1  2⎤
// ⎣3  4⎦
```

### Transpose

Instead of T(), we have Transpose(), which is more compatible with the new function.
This function is useful when the transposed matrix is used as a receiver as follow:

```go
XTX := X.Transpose().Product(X)
```

Note that due to implementation constraints, DenseCopyOf is performed.

## How to work

This adapter embed an adaptee struct.
When a function is called, the adapter prepare new receiver and run original function.
The receiver as the result is wrapped and returned.

```go
type Dense struct {
	*mat.Dense
}

func (m *Dense) Add(b mat.Matrix) *Dense {
	var dense mat.Dense
	dense.Add(m.Dense, b)
	return &Dense{&dense}
}
```


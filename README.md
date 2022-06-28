# Gonum matrix adapter

The light adapter package provides method signatures that allow intuitive operation with fewer lines of code for Gonum matrix.

## Example

Comparison of implementaion for normal equation of Ridge regresssion ($\hat{\theta} = (X^{\top}X + \lambda I)^{-1} X^{\top}Y$).

### Gonum matrix adapter

```go
	X, Y := syntheticData(N, theta)

	I := mat.NewDiagDense(3, []float64{1.0, 1.0, 1.0})
	reg := DenseCopyOf(I).Scale(lambda)

	XTXinv, _ := X.Transpose().Product(X).Add(reg).Inverse()
	XTY := X.Transpose().Product(Y)

	estimated := XTXinv.Product(XTY)
```

### Gonum matrix

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

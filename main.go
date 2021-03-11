package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"gopkg.in/cheggaaa/pb.v1"
	. "gorgonia.org/golgi"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func fizzbuzz(n int) []float64 {
	if n%15 == 0 {
		return []float64{0, 0, 0, 1}
	}
	if n%5 == 0 {
		return []float64{0, 0, 1, 0}
	}
	if n%3 == 0 {
		return []float64{0, 1, 0, 0}
	}
	return []float64{1, 0, 0, 0}
}

func bin(n int) []float64 {
	var r [10]float64
	for d := 0; d < 10; d++ {
		r[d] = float64(n >> d & 1)
	}
	return r[:]
}

func dec(v []float64, n int) string {
	m := v[0]
	j := 0
	for i, vv := range v {
		if m < vv {
			j = i
			m = vv
		}
	}
	switch j {
	case 1:
		return "Fizz"
	case 2:
		return "Buzz"
	case 3:
		return "FizzBuzz"
	}
	return fmt.Sprint(n)
}

func softmax(a *gorgonia.Node) (*gorgonia.Node, error) { return gorgonia.SoftMax(a) }

func main() {
	var epochs, bs int
	flag.IntVar(&epochs, "epochs", 300, "number of epoch")
	flag.IntVar(&bs, "batchsize", 50, "size of batch")
	flag.Parse()

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithName("X"), gorgonia.WithShape(bs, 10), gorgonia.WithInit(gorgonia.GlorotU(1)))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithName("Y"), gorgonia.WithShape(bs, 4), gorgonia.WithInit(gorgonia.GlorotU(1)))
	nn, err := ComposeSeq(x,
		L(ConsFC, WithSize(bs), WithActivation(gorgonia.Rectify)),
		L(ConsFC, WithSize(4), WithActivation(softmax)),
	)
	if err != nil {
		log.Fatal(err)
	}
	out := nn.Fwd(x)
	if err = gorgonia.CheckOne(out); err != nil {
		log.Fatal(err)
	}

	cost := gorgonia.Must(RMS(out, y))
	model := nn.Model()
	if _, err = gorgonia.Grad(cost, model...); err != nil {
		log.Fatal(err)
	}

	m := gorgonia.NewTapeMachine(nn.Graph())
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.001))

	xB := make([]float64, bs*10)
	yB := make([]float64, bs*4)
	for i := 0; i < bs; i++ {
		copy(xB[i*10:], bin(i+121))
		copy(yB[i*4:], fizzbuzz(i+121))
	}

	bar := pb.New(epochs)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	bar.Set(0)
	bar.Start()

	for i := 0; i < epochs; i++ {
		xT := tensor.New(tensor.WithShape(bs, 10), tensor.WithBacking(xB))
		err = gorgonia.Let(x, xT)
		if err != nil {
			log.Fatal(err)
		}

		yT := tensor.New(tensor.WithShape(bs, 4), tensor.WithBacking(yB))
		err = gorgonia.Let(y, yT)
		if err != nil {
			log.Fatal(err)
		}

		if err := m.RunAll(); err != nil {
			log.Fatal(err)
		}
		if err = solver.Step(gorgonia.NodesToValueGrads(nn.Model().Nodes())); err != nil {
			log.Fatal(err)
		}
		m.Reset()

		bar.Increment()
		bar.Update()
	}
	bar.Finish()

	err = save(nn.Model().Nodes())
	if err != nil {
		log.Fatal(err)
	}
}

func save(nodes []*gorgonia.Node) error {
	f, err := os.Create("backup.weights")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, node := range nodes {
		err := enc.Encode(node.Value())
		if err != nil {
			return err
		}
	}
	return nil
}

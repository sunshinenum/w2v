package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"gonpy"
)

type w2v struct {
	embeddings [][]float64
	weights    [][]float64
	bias       []float64
}

// init model params
func build(conf params, wordsCt int, HisEmbeddingsMap map[int][]float64) w2v {
	log.Println("init model ...")
	model := w2v{}
	model.embeddings = make([][]float64, wordsCt)
	for i := 0; i < wordsCt; i++ {
		vec, ex := HisEmbeddingsMap[i]
		if ex {
			model.embeddings[i] = vec
		} else {
			model.embeddings[i] = make([]float64, conf.EmbeddingSz)
			for j := range model.embeddings[i] {
				model.embeddings[i][j] = -0.025 + rand.Float64()*0.05
			}
		}
	}
	model.weights = make([][]float64, wordsCt)
	for i := 0; i < wordsCt; i++ {
		vec, ex := HisEmbeddingsMap[i]
		if ex {
			model.weights[i] = vec
		} else {
			model.weights[i] = make([]float64, conf.EmbeddingSz)
			for j := range model.weights[i] {
				model.weights[i][j] = -0.025 + rand.Float64()*0.05
			}
		}
	}
	model.bias = make([]float64, wordsCt)
	for i := range model.bias {
		model.bias[i] = -0.025 + rand.Float64()*0.05
	}
	return model
}

func trainMaster(model w2v, conf params, trainData [][]int, probArr []int) {
	log.Println("start training ...")
	samplesSingleThread := len(trainData) / conf.Threads
	left := len(trainData) % conf.Threads
	wg := &sync.WaitGroup{}
	wg.Add(conf.Threads)
	for i := 0; i < conf.Threads; i++ {
		if i != conf.Threads-1 {
			go train(model, conf, trainData[i*samplesSingleThread:(i+1)*samplesSingleThread], probArr, i, wg)
		} else {
			go train(model, conf, trainData[i*samplesSingleThread:(i+1)*samplesSingleThread+left], probArr, i, wg)
		}
	}
	wg.Wait()
}

func train(model w2v, conf params, trainData [][]int, probArr []int, pid int, wg *sync.WaitGroup) {
	defer wg.Done()
	threadRnd := rand.New(rand.NewSource(time.Now().Unix()))
	for iterCu := 0; iterCu < conf.Iters; iterCu++ {
		pLoss, nLoss, reg, lossCu, regCu, posSamples, negSamples := 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		start := time.Now()
		for _, words := range trainData {
			// 根据Session长度过滤语料
			if len(words) < conf.MinWords {
				continue
			}
			for i, x := range words {
				for j := i - conf.WindowSz; j < i+conf.WindowSz; j++ {
					if j < 0 || j == i || j >= len(words) {
						continue
					}
					// positive
					lossCu, regCu = trainSample(model, conf, x, words[j], 1.0, conf.EmbeddingReused)
					pLoss += lossCu
                    reg += regCu
                    posSamples += 1.0
					// negative
					for negCu := 0; negCu < conf.Negatives; negCu++ {
						yNeg := probArr[threadRnd.Intn(len(probArr))]
						lossCu, regCu = trainSample(model, conf, x, yNeg, 0.0, conf.EmbeddingReused)
						nLoss += lossCu
                        reg += regCu
                        negSamples += 1.0
					}
				}
			}
		}
		final := time.Now()
		if pid == 0 {
			fmt.Printf("iter: %2d ploss: %.4f nloss: %.4f reg: %.4f lr: %.4f time_cost: %.2f\n",
				iterCu, pLoss/posSamples, nLoss/negSamples, reg/(posSamples+negSamples), conf.Lr, final.Sub(start).Seconds())
		}
		start = final
		conf.Lr = conf.LrDownRate * conf.Lr
	}
}

func trainSample(model w2v, conf params, x int, y int, l float64, reuse int) (float64, float64) {
	/*
	   trainSample: train a single sample
	   x: iid
	   y: iid
	   l: label
	   reuse: x, y in sample space

	   return: mf loss, l2 loss
	*/
	s := 0.0
	for i, v := range model.embeddings[x] {
		if reuse == 1 {
			s += model.embeddings[y][i] * v
		} else {
			s += model.weights[y][i] * v
		}
	}
	// probability
	s = 1.0 / (1.0 + math.Exp(-s))
	if reuse == 1 {
		for i, v := range model.embeddings[x] {
			model.embeddings[x][i] -= conf.Lr * ((s-l)*model.embeddings[y][i] + conf.RegRate*v)
		}
		for i, v := range model.embeddings[y] {
			model.embeddings[y][i] -= conf.Lr * ((s-l)*model.embeddings[x][i] + conf.RegRate*v)
		}
	} else {
		for i, v := range model.embeddings[x] {
			model.embeddings[x][i] -= conf.Lr * ((s-l)*model.weights[y][i] + conf.RegRate*v)
		}
		for i, v := range model.weights[y] {
			model.weights[y][i] -= conf.Lr * ((s-l)*model.embeddings[x][i] + conf.RegRate*v)
		}
	}
	// cal log loss & l2 loss
	loss := -(l*math.Log(s) + (1.0-l)*math.Log(1.0-s))
	l2 := 0.0
	for _, v := range model.embeddings[x] {
		l2 += v * v
	}
	l2 = math.Sqrt(l2) / float64(conf.EmbeddingSz)
	return loss, l2
}

func l2Norm(vec []float64) []float64 {
	sum := 0.0
	ret := make([]float64, len(vec))
	for _, v := range vec {
		sum += v * v
	}
	if sum == 0.0 {
		return ret
	}
	sum = math.Sqrt(sum)
	for i, _ := range vec {
		ret[i] = vec[i] / sum
	}
	return ret
}

func save(model w2v, conf params) {
	// new a writer
	wtr, _ := gonpy.NewFileWriter(conf.EmbeddingPath)
	wtr.Shape = []int{len(model.embeddings), conf.EmbeddingSz}
	wtr.Version = 2
	// new a 1d array
	embedding := make([]float64, 0)
	log.Println("normalize the embedding with l2 ...")
	for _, line := range model.embeddings {
		line = l2Norm(line)
		embedding = append(embedding, line...)
	}
	log.Println("write out the embedding ...")
	wtr.WriteFloat64(embedding)
}

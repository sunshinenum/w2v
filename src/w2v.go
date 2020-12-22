package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/kshedden/gonpy"
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
		pLoss := 0.0
		nLoss := 0.0
		reg := 0.0
		lossCu := 0.0
		regCu := 0.0
		posSamples := 0.0
		negSamples := 0.0
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
					if conf.EmbeddingReused == 0 {
						lossCu, regCu = trainSample(model, conf, x, words[j], 1.0)
					} else {
						lossCu, regCu = trainSampleReuseEmbedding(model, conf, x, words[j], 1.0)
					}
					pLoss += lossCu
					reg += regCu
					posSamples += 1.0
					// negative
					for negCu := 0; negCu < conf.Negatives; negCu++ {
						yNeg := probArr[threadRnd.Intn(len(probArr))]
						if conf.EmbeddingReused == 0 {
							lossCu, regCu = trainSample(model, conf, x, yNeg, 0.0)
						} else {
							lossCu, regCu = trainSampleReuseEmbedding(model, conf, x, yNeg, 0.0)
						}
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

func trainSample(model w2v, conf params, x int, y int, l float64) (float64, float64) {
	s := 0.0
	for i, v := range model.embeddings[x] {
		s += model.weights[y][i] * v
	}
	s = 1.0 / (1.0 + math.Exp(-s))
	// for prevent loss from NaN
	if s < 1e-8 {
		s = 1e-8
	} else if s > 1-1e-8 {
		s = 1 - 1e-8
	}
	e := s - l
	// update
	for i, v := range model.embeddings[x] {
		model.embeddings[x][i] -= conf.Lr * (e*model.weights[y][i] + conf.RegRate*v)
		// 正则
		// if model.embeddings[x][i] > conf.MaxVal {
		// 	model.embeddings[x][i] = conf.MaxVal
		// } else if model.embeddings[x][i] < -conf.MaxVal {
		// 	model.embeddings[x][i] = -conf.MaxVal
		// }
	}
	for i, v := range model.weights[x] {
		model.weights[y][i] -= conf.Lr * (e*model.embeddings[x][i] + conf.RegRate*v)
		// 正则
		// if model.weights[y][i] > conf.MaxVal {
		// 	model.weights[y][i] = conf.MaxVal
		// } else if model.weights[y][i] < -conf.MaxVal {
		// 	model.weights[y][i] = -conf.MaxVal
		// }
	}
	// loss & reg
	loss := -(l*math.Log(s) + (1.0-l)*math.Log(1.0-s))
	reg := 0.0
	for i, v := range model.embeddings[x] {
		reg += v * v
		reg += model.weights[y][i] * model.weights[y][i]
	}
	reg = math.Sqrt(reg/2) / float64(conf.EmbeddingSz)
	return loss, reg
}

// reuse embedding
func trainSampleReuseEmbedding(model w2v, conf params, x int, y int, l float64) (float64, float64) {
	s := 0.0
	for i, v := range model.embeddings[x] {
		s += model.embeddings[y][i] * v
	}
	s = 1.0 / (1.0 + math.Exp(-s))
	// for prevent loss from NaN
	if s < 1e-8 {
		s = 1e-8
	} else if s > 1-1e-8 {
		s = 1 - 1e-8
	}
	e := s - l
	// update
	for i, v := range model.embeddings[x] {
		model.embeddings[x][i] -= conf.Lr * (e*model.embeddings[y][i] + conf.RegRate*v)
		// if model.embeddings[x][i] > conf.MaxVal {
		// 	model.embeddings[x][i] = conf.MaxVal
		// } else if model.embeddings[x][i] < -conf.MaxVal {
		// 	model.embeddings[x][i] = -conf.MaxVal
		// }
	}
	for i, v := range model.embeddings[x] {
		model.embeddings[y][i] -= conf.Lr * (e*model.embeddings[x][i] + conf.RegRate*v)
		// if model.embeddings[y][i] > conf.MaxVal {
		// 	model.embeddings[y][i] = conf.MaxVal
		// } else if model.embeddings[y][i] < -conf.MaxVal {
		// 	model.embeddings[y][i] = -conf.MaxVal
		// }
	}
	// loss & reg
	loss := -(l*math.Log(s) + (1.0-l)*math.Log(1.0-s))
	reg := 0.0
	for i, v := range model.embeddings[x] {
		reg += v * v
		reg += model.embeddings[y][i] * model.embeddings[y][i]
	}
	reg = math.Sqrt(reg/2) / float64(conf.EmbeddingSz)
	return loss, reg
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
	wtr, _ := gonpy.NewFileWriter(conf.EmbeddingPath)
	wtr.Shape = []int{len(model.embeddings), conf.EmbeddingSz}
	wtr.Version = 2
	embedding := make([]float64, 0)
	log.Println("normalize the embedding with l2 ...")
	for _, line := range model.embeddings {
		line = l2Norm(line)
		embedding = append(embedding, line...)
	}
	log.Println("write out the embedding ...")
	wtr.WriteFloat64(embedding)
}

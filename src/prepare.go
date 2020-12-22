package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/kshedden/gonpy"
)

// read kv to map
func loadKVtoMap(filePath string) (map[string]int, []string) {
	ip, _ := os.Open(filePath)
	defer ip.Close()
	scanner := bufio.NewScanner(ip)
	line := ""
	wordsCount := make(map[string]int)
	wordsHis := make([]string, 0)
	for scanner.Scan() {
		line = scanner.Text()
		secs := strings.Split(line, "\t")
		word := secs[0]
		count, _ := strconv.ParseInt(secs[1], 0, 64)
		wordsCount[word] = int(count)
		wordsHis = append(wordsHis, word)
	}
	if err := scanner.Err(); err != nil {
		fmt.Printf("Err while load %s\n", filePath)
	}
	return wordsCount, wordsHis
}

func new2DSliceF64(a int, b int) [][]float64 {
	ret := make([][]float64, a)
	for i := 0; i < a; i++ {
		ret[i] = make([]float64, b)
	}
	return ret
}

func reshapeSlice(vec []float64, shape []int) [][]float64 {
	if len(vec) != int(shape[0]*shape[1]) {
		fmt.Println("reshape failed !")
		os.Exit(-1)
	}
	ret := make([][]float64, shape[0])
	x := int(shape[0])
	y := int(shape[1])
	for i := 0; i < x; i++ {
		ret[i] = vec[y*i : y*(i+1)]
	}
	return ret
}

// load embeddings from numpy array
func loadNpy(path string) [][]float64 {
	r, _ := gonpy.NewFileReader(path)
	data, _ := r.GetFloat64()
	ret := reshapeSlice(data, r.Shape)
	return ret
}

func prepare(w2vParams params) ([][]int, int, []int, map[int][]float64) {
	words := make([]string, 0)
	wordsCt := make(map[string]int)

	log.Println("loading train data ...")
	dat, _ := ioutil.ReadFile(w2vParams.TrainPath)
	lines := strings.Split(string(dat), "\n")
	lineWords := make([][]string, 0)
	tokens := 0
	for i, _ := range lines {
		if len(lines[i]) == 0 {
			continue
		}
		secs := strings.Split(lines[i], " ")
		for j := 0; j < len(secs); j++ {
			_, e := wordsCt[secs[j]]
			if !e {
				wordsCt[secs[j]] = 1
				words = append(words, secs[j])
			} else {
				wordsCt[secs[j]]++
			}
			tokens++
		}
		lineWords = append(lineWords, secs)
	}
	log.Printf("%d lines, %d words, %d tokens loaded.", len(lines), len(words), tokens)

	// 如果增量训练，加载词的历史出现次数
	wordsCountHis := make(map[string]int)
	wordsHis := make([]string, 0)
	if w2vParams.IncreaseTraining != 0 {
		// 加载词频信息
		wordsCountHis, wordsHis = loadKVtoMap(w2vParams.WordsPath)
		for k, v := range wordsCountHis {
			_, ex := wordsCt[k]
			if ex {
				wordsCt[k] = v + wordsCt[k]
			} else {
				wordsCt[k] = v
				words = append(words, k)
			}
		}
	}

	// 根据词频信息排序
	log.Println("sort words by words count...")
	sort.Slice(words, func(i, j int) bool {
		return wordsCt[words[i]] > wordsCt[words[j]]
	})

	// 过滤低频词，生成词表索引
	log.Println("filter low freq words ...")
	wordsFtd := make([]string, 0)
	wordsIndex := make(map[string]int)
	index := 0
	for _, w := range words {
		wc, ex := wordsCt[w]
		if ex && wc >= w2vParams.WordsMinCount {
			wordsFtd = append(wordsFtd, w)
			wordsIndex[w] = index
			index++
		}
	}

	// 如果增量更新，加载embedding
	HisEmbeddingsMap := make(map[int][]float64)
	if w2vParams.IncreaseTraining != 0 {
		log.Println("increase training load embedding ...")
		embedding := loadNpy(w2vParams.EmbeddingPath)
		if len(embedding) != len(wordsHis) {
			log.Println("增量更新失败，len(embedding) != len(wordsHis) !")
			os.Exit(-1)
		}
		for i, word := range wordsHis {
			index, ex := wordsIndex[word]
			if ex {
				HisEmbeddingsMap[index] = embedding[i]
			}
		}
	}

	log.Println("index train data ...")
	trainDataIndexed := make([][]int, 0)
	for _, line := range lines {
		secs := strings.Split(line, " ")
		secIndexedCu := make([]int, 0)
		for _, sec := range secs {
			index, ex := wordsIndex[sec]
			if ex {
				secIndexedCu = append(secIndexedCu, index)
			}
		}
		if len(secIndexedCu) > 0 {
			trainDataIndexed = append(trainDataIndexed, secIndexedCu)
		}
	}

	log.Println("save words & index & counts & prob ...")
	op, _ := os.Create(w2vParams.WordsPath)
	probArr := make([]int, 0)
	filtered_words := 0
	for i, key := range words {
		wc, ex := wordsCt[key]
		if ex && wc < w2vParams.WordsMinCount {
			filtered_words++
			continue
		}
		freq := int(math.Pow(float64(wordsCt[key]), w2vParams.NegativeE))
		line := fmt.Sprintf("%s\t%d\t%d\t%d\n", key, i, wordsCt[key], freq)
		op.WriteString(line)
		arr := make([]int, freq)
		for j := range arr {
			arr[j] = int(i)
		}
		probArr = append(probArr, arr...)
	}
	op.Close()
	log.Printf("%d words filtered by words_min_count: %d", filtered_words, w2vParams.WordsMinCount)

	return trainDataIndexed, len(wordsIndex), probArr, HisEmbeddingsMap
}

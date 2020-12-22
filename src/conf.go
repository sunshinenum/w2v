package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

type params struct {
	Lr               float64 `json:"lr"`
	RegRate          float64 `json:"reg_rate"`
	LrDownRate       float64 `json:"lr_down_rate"`
	MaxVal           float64 `json:"max_val"`
	EmbeddingSz      int     `json:"embedding_sz"`
	MinWords         int     `json:"min_words"`
	WordsMinCount    int     `json:"words_min_count"`
	WindowSz         int     `json:"window_sz"`
	Negatives        int     `json:"negatives"`
	NegativeE        float64 `json:"negative_e"`
	EmbeddingReused  int     `json:"embedding_reused"`
	IncreaseTraining int     `json:"increase_training"`
	Iters            int     `json:"iters"`
	Threads          int     `json:"threads"`
	ModelType        int     `json:"model_type"`
	TrainPath        string  `json:"train_path"`
	EmbeddingPath    string  `json:"embedding_path"`
	WordsPath        string  `json:"words_path"`
}

func loadConf(confPath string) params {
	w2vConf := params{}
	file, _ := ioutil.ReadFile(confPath)
	err := json.Unmarshal([]byte(file), &w2vConf)
	if err == nil {
		fmt.Println("load conf success!", w2vConf)
	} else {
		fmt.Println("load conf error!", w2vConf)
	}
	return w2vConf
}

// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// WeightsFileName is the name of the bf16 safetensors checkpoint inside the
// HuggingFace repository.
const WeightsFileName = "sam3.1_multiplex_bf16.safetensors"

// Model holds the configuration and reference to the HuggingFace repository.
type Model struct {
	Repo   *hub.Repo
	Config *Config

	totalParameters *int64
	totalBytes      *int64
}

// LoadModel loads the configuration and returns a Model.
//
// The official checkpoint does not ship a usable config.json; the architecture
// is fully described by the reference implementation, so [DefaultConfig] is
// used. If the repository happens to contain a config.json it is used instead.
func LoadModel(repo *hub.Repo) (*Model, error) {
	m := &Model{Repo: repo, Config: DefaultConfig()}

	path, err := repo.DownloadFile("config.json")
	if err == nil {
		if c, cerr := LoadConfig(path); cerr == nil {
			m.Config = c
		}
	}

	return m, nil
}

// loadCastToFloat32 converts a tensor to float32 (used to unify the mixed
// F32/BF16 checkpoint dtypes). The compiled graph is shared and reused across
// all conversions.
func loadCastToFloat32(backend compute.Backend, t *tensors.Tensor) (*tensors.Tensor, error) {
	exec, err := graph.NewExec(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Float32)
	})
	if err != nil {
		return nil, err
	}
	res, err := exec.Call(t)
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

// mapTensorName translates a safetensors name (e.g.
// "detector.transformer.encoder.layers.0.linear1.weight") into a GoMLX scope
// path and variable name. It returns ok=false for tensors that should be
// skipped (tracker/video weights, RoPE buffers, and the unused text projection).
func mapTensorName(name string) (scopePath []string, varName string, ok bool) {
	if !strings.HasPrefix(name, "detector.") {
		return nil, "", false
	}
	// The image model does not use the interactive/propagation necks, the
	// tracker stack, precomputed RoPE buffers, or the pooled text projection.
	if strings.Contains(name, ".interactive_convs.") ||
		strings.Contains(name, ".propagation_convs.") ||
		strings.HasSuffix(name, "freqs_cis") ||
		strings.HasSuffix(name, "text_projection") {
		return nil, "", false
	}

	stripped := strings.TrimPrefix(name, "detector.")
	parts := strings.Split(stripped, ".")
	if len(parts) < 1 {
		return nil, "", false
	}
	varName = parts[len(parts)-1]
	scopePath = parts[:len(parts)-1]
	return scopePath, varName, true
}

// LoadStore loads the safetensors weights into the GoMLX model.Store.
//
// Weights are stored using their PyTorch names and layouts (e.g. a linear layer
// weight has shape [out, in]); the graph builders in this package read those
// variables and apply the required transposes on the fly.
func (m *Model) LoadStore(backend compute.Backend, store *model.Store) error {
	var totalParams int64
	var totalBytes int64

	reader, err := safetensors.NewEmpty(m.Repo).NewTensorReader(WeightsFileName)
	if err != nil {
		return errors.Wrap(err, "failed to open safetensors checkpoint")
	}
	defer reader.Close()

	names := mapsKeys(reader.Header.Tensors)
	slices.Sort(names)

	for tensorAndName, err := range reader.IterTensors(backend, names) {
		if err != nil {
			return errors.WithMessagef(err, "failed loading variables of model %q", m.Repo.ID)
		}

		scopePath, varName, ok := mapTensorName(tensorAndName.Name)
		if !ok {
			klog.V(1).Infof("Skipping unmapped tensor: %s\n", tensorAndName.Name)
			tensorAndName.Tensor.FinalizeAll()
			continue
		}

		tensorToLoad := tensorAndName.Tensor
		if tensorToLoad.DType() != dtypes.Float32 {
			// Unify the mixed F32/BF16 checkpoint to float32.
			converted, err := loadCastToFloat32(backend, tensorToLoad)
			if err != nil {
				tensorToLoad.FinalizeAll()
				return errors.WithMessagef(err, "failed to cast tensor %q to float32", tensorAndName.Name)
			}
			tensorToLoad.FinalizeAll()
			tensorToLoad = converted
		}

		shape := tensorToLoad.Shape()
		totalParams += int64(shape.Size())
		totalBytes += int64(shape.ByteSize())

		subScope := store.RootScope()
		for _, subScopeName := range scopePath {
			subScope = subScope.In("%s", subScopeName)
		}
		subScope.VariableWithValue(varName, tensorToLoad)
	}

	klog.V(1).Infof("Loaded %s parameters (%s bytes)",
		humanize.Count(totalParams), humanize.Bytes(totalBytes))
	m.totalParameters = &totalParams
	m.totalBytes = &totalBytes

	return nil
}

func mapsKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

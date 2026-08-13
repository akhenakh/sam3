// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	c := DefaultConfig()
	require.Equal(t, 1008, c.ImageSize)
	require.Equal(t, 1024, c.Backbone.EmbedDim)
	require.Equal(t, 32, c.Backbone.Depth)
	require.Equal(t, 24, c.Backbone.WindowSize)
	require.Equal(t, 200, c.Transformer.NumQueries)
	require.Equal(t, 32, c.Text.ContextLength)
	require.Equal(t, 49408, c.Text.VocabSize)
}

func TestTokenizer(t *testing.T) {
	// The BPE vocabulary ships with the reference Python package; skip if not
	// available in this checkout.
	const bpePath = "../../sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
	tok, err := newTokenizer(bpePath)
	if err != nil {
		t.Skipf("BPE vocab not available: %v", err)
	}

	ids := tok.tokenize("buildings")
	require.Len(t, ids, ContextLength)
	require.Equal(t, tok.sotID, ids[0])
	// The last non-zero token is the end-of-text token.
	last := 0
	for _, v := range ids {
		if v != 0 {
			last = v
		}
	}
	require.Equal(t, tok.eotID, last)
	require.Equal(t, 0, ids[ContextLength-1]) // "buildings" fits within the context.
}

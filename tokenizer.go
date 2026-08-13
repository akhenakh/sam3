// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"compress/gzip"
	"fmt"
	"html"
	"os"
	"regexp"
	"strings"
)

// ContextLength is the maximum number of text tokens (including the start/end
// of text tokens) accepted by the SAM 3.1 text encoder.
const ContextLength = 32

const (
	sotToken = "<start_of_text>"
	eotToken = "<end_of_text>"
)

// tokenizer implements the CLIP-style byte-pair-encoding tokenizer used by the
// SAM 3.1 text encoder. It mirrors the reference `SimpleTokenizer`.
type tokenizer struct {
	byteEncoder map[byte]string
	encoder     map[string]int
	bpeRanks    map[[2]string]int
	pat         *regexp.Regexp
	sotID       int
	eotID       int
}

// newTokenizer loads the BPE vocabulary from a gzipped merge file (the SAM 3
// `bpe_simple_vocab_16e6.txt.gz` asset).
func newTokenizer(path string) (*tokenizer, error) {
	merges, err := loadMerges(path)
	if err != nil {
		return nil, err
	}

	t := &tokenizer{
		byteEncoder: bytesToUnicode(),
		encoder:     make(map[string]int),
		bpeRanks:    make(map[[2]string]int),
	}

	// Build the vocabulary exactly like the reference implementation: the 256
	// byte-to-unicode symbols (in their construction order), then the same 256
	// symbols with the "</w>" suffix, then the merge results, then the two
	// special tokens.
	vocab := make([]string, 0, 49408)
	for b := 0; b < 256; b++ {
		vocab = append(vocab, t.byteEncoder[byte(b)])
	}
	for b := 0; b < 256; b++ {
		vocab = append(vocab, t.byteEncoder[byte(b)]+"</w>")
	}
	for _, m := range merges {
		vocab = append(vocab, m[0]+m[1])
	}
	specials := []string{sotToken, eotToken}
	vocab = append(vocab, specials...)

	for i, v := range vocab {
		t.encoder[v] = i
	}
	for i, m := range merges {
		t.bpeRanks[m] = i
	}

	special := strings.Join(specials, "|")
	t.pat = regexp.MustCompile(
		special + `|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`)
	t.sotID = t.encoder[sotToken]
	t.eotID = t.encoder[eotToken]
	return t, nil
}

// loadMerges reads the BPE merge pairs from a gzipped merge file. The first
// line is a version comment and is skipped; only the first 48894 merges are
// used (matching the reference slice `merges[1:49152-256-2+1]`).
func loadMerges(path string) ([][2]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open BPE vocab file: %w", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("failed to open gzip BPE vocab file: %w", err)
	}
	defer gz.Close()

	data := make([]byte, 0, 1024*1024)
	buf := make([]byte, 64*1024)
	for {
		n, err := gz.Read(buf)
		data = append(data, buf[:n]...)
		if err != nil {
			break
		}
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("BPE vocab file has too few lines")
	}
	const maxMerges = 49152 - 256 - 2 // 48894
	lines = lines[1:]
	if len(lines) > maxMerges {
		lines = lines[:maxMerges]
	}

	merges := make([][2]string, 0, len(lines))
	for _, line := range lines {
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}
		merges = append(merges, [2]string{parts[0], parts[1]})
	}
	return merges, nil
}

// bytesToUnicode reproduces the reference `bytes_to_unicode` table.
func bytesToUnicode() map[byte]string {
	bs := make([]int, 0, 256)
	for b := 33; b <= 126; b++ {
		bs = append(bs, b)
	}
	for b := 161; b <= 172; b++ {
		bs = append(bs, b)
	}
	for b := 174; b <= 255; b++ {
		bs = append(bs, b)
	}
	cs := append([]int{}, bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if !containsInt(bs, b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	m := make(map[byte]string, 256)
	for i := range bs {
		m[byte(bs[i])] = string(rune(cs[i]))
	}
	return m
}

func containsInt(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// basicClean applies the reference `basic_clean` (HTML unescape + strip). The
// ftfy unicode fix is a no-op for the ASCII aerial-imagery prompts.
func basicClean(text string) string {
	return strings.TrimSpace(html.UnescapeString(html.UnescapeString(text)))
}

func whitespaceClean(text string) string {
	re := regexp.MustCompile(`\s+`)
	return strings.TrimSpace(re.ReplaceAllString(text, " "))
}

// cleanFn applies the reference "lower" cleaning: basic clean, whitespace
// collapse, then lower-case.
func cleanFn(text string) string {
	return strings.ToLower(whitespaceClean(basicClean(text)))
}

// getPairs returns the set of adjacent symbol pairs in a word.
func getPairs(word []string) [][2]string {
	pairs := make([][2]string, 0, len(word))
	for i := 0; i+1 < len(word); i++ {
		pairs = append(pairs, [2]string{word[i], word[i+1]})
	}
	return pairs
}

// bpe applies byte-pair encoding to a single token (already mapped through the
// byte encoder), returning space-separated sub-tokens.
func (t *tokenizer) bpe(token string) string {
	runes := []rune(token)
	word := make([]string, 0, len(runes))
	for i := 0; i < len(runes); i++ {
		if i == len(runes)-1 {
			word = append(word, string(runes[i])+"</w>")
		} else {
			word = append(word, string(runes[i]))
		}
	}

	pairs := getPairs(word)
	if len(pairs) == 0 {
		return token + "</w>"
	}

	for {
		// Find the lowest-rank bigram.
		best := -1
		bestRank := int(^uint(0) >> 1)
		for _, p := range pairs {
			r, ok := t.bpeRanks[p]
			if ok && r < bestRank {
				bestRank = r
				best = indexOfPair(word, p)
			}
		}
		if best == -1 {
			break
		}

		first := word[best]
		second := word[best+1]
		newWord := make([]string, 0, len(word))
		i := 0
		for i < len(word) {
			j := indexOf(word, first, i)
			if j == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}
			newWord = append(newWord, word[i:j]...)
			i = j
			if word[i] == first && i < len(word)-1 && word[i+1] == second {
				newWord = append(newWord, first+second)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}
		word = newWord
		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}

	return strings.Join(word, " ")
}

func indexOf(word []string, s string, from int) int {
	for i := from; i < len(word); i++ {
		if word[i] == s {
			return i
		}
	}
	return -1
}

func indexOfPair(word []string, p [2]string) int {
	for i := 0; i+1 < len(word); i++ {
		if word[i] == p[0] && word[i+1] == p[1] {
			return i
		}
	}
	return -1
}

// encode converts a text string into BPE token ids.
func (t *tokenizer) encode(text string) []int {
	text = cleanFn(text)
	tokens := t.pat.FindAllString(text, -1)

	ids := make([]int, 0, 16)
	for _, token := range tokens {
		var sb strings.Builder
		for _, b := range []byte(token) {
			sb.WriteString(t.byteEncoder[b])
		}
		encoded := t.bpe(sb.String())
		for _, sub := range strings.Split(encoded, " ") {
			ids = append(ids, t.encoder[sub])
		}
	}
	return ids
}

// tokenize returns the padded token-id sequence [contextLength] for a text
// prompt: <sot> tokens <eot> padded with zeros.
func (t *tokenizer) tokenize(text string) []int {
	ids := []int{t.sotID}
	ids = append(ids, t.encode(text)...)
	ids = append(ids, t.eotID)
	if len(ids) > ContextLength {
		ids = ids[:ContextLength]
		ids[len(ids)-1] = t.eotID
	}
	out := make([]int, ContextLength)
	copy(out, ids)
	return out
}

// valid reports whether the token sequence has at least one non-zero token.
func validTokenIds(ids []int) bool {
	for _, v := range ids {
		if v != 0 {
			return true
		}
	}
	return false
}

// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam3

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
)

// bilinearSample gathers img (channels-first [batch, C, H, W]) at fractional
// coordinates ys/xs (both shaped [batch, ...]) using bilinear interpolation with
// edge clamping. It returns [batch, C, ...].
func bilinearSample(img, ys, xs *Node) *Node {
	g := img.Graph()
	dims := img.Shape().Dimensions
	b, c, h, w := dims[0], dims[1], dims[2], dims[3]

	// Flatten the trailing coordinate axes to [batch, K].
	coordShape := ys.Shape().Dimensions
	k := 1
	for _, d := range coordShape[1:] {
		k *= d
	}
	ys = Reshape(ys, b, k)
	xs = Reshape(xs, b, k)

	hMax := Scalar(g, xs.DType(), float64(h-1))
	wMax := Scalar(g, xs.DType(), float64(w-1))
	zero := Scalar(g, xs.DType(), 0.0)

	yc := Clamp(zero, ys, hMax)
	xc := Clamp(zero, xs, wMax)

	y0 := Floor(yc)
	x0 := Floor(xc)
	wy := Sub(yc, y0)
	wx := Sub(xc, x0)

	y0i := ConvertDType(y0, dtypes.Int32)
	x0i := ConvertDType(x0, dtypes.Int32)
	one := Scalar(g, dtypes.Int32, 1)
	y1i := Min(Add(y0i, one), Scalar(g, dtypes.Int32, int32(h-1)))
	x1i := Min(Add(x0i, one), Scalar(g, dtypes.Int32, int32(w-1)))

	imgFlat := Reshape(img, b*c*h*w) // rank 1

	fullIndex := func(yi, xi *Node) *Node {
		pix := Add(Mul(yi, Scalar(g, dtypes.Int32, int32(w))), xi) // [batch, k]
		bIdx := MulScalar(IotaFull(g, shapes.Make(dtypes.Int32, b, 1, 1)), int32(c*h*w))
		cIdx := MulScalar(IotaFull(g, shapes.Make(dtypes.Int32, 1, c, 1)), int32(h*w))
		pix = ExpandAxes(pix, 1) // [batch, 1, k]
		full := Add(Add(bIdx, cIdx), pix)
		return ExpandAxes(full, -1) // [batch, c, k, 1]
	}

	v00 := Gather(imgFlat, fullIndex(y0i, x0i)) // [batch, c, k]
	v01 := Gather(imgFlat, fullIndex(y0i, x1i))
	v10 := Gather(imgFlat, fullIndex(y1i, x0i))
	v11 := Gather(imgFlat, fullIndex(y1i, x1i))

	// Weights [batch, 1, k].
	wy = ExpandAxes(wy, 1)
	wx = ExpandAxes(wx, 1)
	ony := OneMinus(wy)
	onx := OneMinus(wx)

	out := Add(
		Mul(v00, Mul(ony, onx)),
		Add(
			Mul(v01, Mul(ony, wx)),
			Add(Mul(v10, Mul(wy, onx)), Mul(v11, Mul(wy, wx))),
		),
	)

	// Reshape back to [batch, c, ...].
	outDims := append([]int{b, c}, coordShape[1:]...)
	return Reshape(out, outDims...)
}

// posEncode1D computes the sine/cosine encoding of a coordinate (as in
// PositionEmbeddingSine._encode_xy), returning [..., half].
func posEncode1D(g *Graph, coord *Node, half int, temperature float64) *Node {
	scale := 2 * math.Pi
	dimT := IotaFull(g, shapes.Make(dtypes.Float32, half))
	dimT = Pow(ConstAs(dimT, temperature), DivScalar(MulScalar(Floor(DivScalar(dimT, 2.0)), 2.0), float64(half)))

	x := Div(ExpandAxes(MulScalar(coord, scale), -1), Reshape(dimT, 1, 1, half))
	return interleaveSinCos(x)
}

// geometryEncoder runs the SequenceGeometryEncoder on point/box prompts.
//
// points [nPoints, batch, 2] (normalized xy), pointLabels [nPoints, batch]
// (int32), pointMask [batch, nPoints] (bool, true = pad); boxes [nBoxes, batch,
// 4] (normalized cxcywh), boxLabels [nBoxes, batch], boxMask [batch, nBoxes].
// imgFeat is the last FPN level in sequence-first layout [H*W, batch, 256] and
// imgFeatSpatial is the same in [batch, 256, H, W] layout (used for pooling).
//
// Returns geoFeats [seq, batch, 256] and geoMask [batch, seq].
func geometryEncoder(scope *model.Scope, points, pointLabels, pointMask, boxes, boxLabels, boxMask, imgFeat, imgFeatSpatial, imgPos *Node, cfg *Config) (geoFeats, geoMask *Node) {
	g := points.Graph()
	bs := points.Shape().Dim(1)
	dModel := cfg.Geometry.DModel
	half := dModel / 2

	// Image features for pooling are pre-normalized, then reshaped spatially.
	imgPreNorm := layerNormLast(scope.In("img_pre_norm"), imgFeat, 1e-5) // [HW, batch, 256]
	spatialDims := imgFeatSpatial.Shape().Dimensions
	h, w := spatialDims[2], spatialDims[3]
	imgForPool := Reshape(TransposeAllAxes(imgPreNorm, 1, 2, 0), bs, dModel, h, w) // [batch, 256, H, W]

	// Encode points.
	pointsEmb := encodePoints(scope, points, pointLabels, imgForPool, dModel, half)
	// Encode boxes.
	boxesEmb := encodeBoxes(scope, boxes, boxLabels, imgForPool, dModel, half)

	// Concatenate [points; boxes; cls].
	cls := scope.In("cls_embed").GetVariable("weight").NodeValue(g) // [1, dModel]
	cls = BroadcastToDims(ExpandAxes(cls, 1), 1, bs, dModel)

	geoFeats = Concatenate([]*Node{pointsEmb, boxesEmb, cls}, 0)
	zeroMask := Zeros(g, shapes.Make(dtypes.Bool, bs, 1))
	geoMask = Concatenate([]*Node{pointMask, boxMask, zeroMask}, 1)

	// Final projection + norm.
	geoFeats = layerNormLast(scope.In("norm"), linear(scope.In("final_proj"), geoFeats), 1e-5)

	// Transformer encoder layers (self-attention over prompt + cross-attention
	// to image).
	encodeScope := scope.In("encode")
	for i := range cfg.Geometry.NumLayers {
		geoFeats = geometryEncoderLayer(encodeScope.In("%d", i), geoFeats, geoMask, imgFeat, imgPos, dModel)
	}
	geoFeats = layerNormLast(scope.In("encode_norm"), geoFeats, 1e-5)
	return geoFeats, geoMask
}

// geometryEncoderLayer runs one TransformerEncoderLayer of the geometry encoder.
// tgt is [seq, batch, dModel] (seq-first), keyPaddingMask [batch, seq].
func geometryEncoderLayer(scope *model.Scope, tgt, keyPaddingMask, memory, pos *Node, dModel int) *Node {
	tgt2 := layerNormLast(scope.In("norm1"), tgt, 1e-5)
	mask := keyPaddingMaskToAttention(keyPaddingMask, 8)
	self := multiHeadAttention(scope.In("self_attn"), tgt2, tgt2, tgt2, 8, false, mask)
	tgt = Add(tgt, self)

	tgt2 = layerNormLast(scope.In("norm2"), tgt, 1e-5)
	k := Add(memory, pos)
	cross := multiHeadAttention(scope.In("cross_attn_image"), tgt2, k, memory, 8, false, nil)
	tgt = Add(tgt, cross)

	tgt2 = layerNormLast(scope.In("norm3"), tgt, 1e-5)
	ff := linear(scope.In("linear1"), tgt2)
	ff = activation.Relu(ff)
	ff = linear(scope.In("linear2"), ff)
	return Add(tgt, ff)
}

// encodePoints builds the point prompt embeddings (direct projection + pooling
// + positional encoding, summed) plus the label embedding.
func encodePoints(scope *model.Scope, points, labels, imgForPool *Node, dModel, half int) *Node {
	g := points.Graph()

	var emb *Node

	// Direct projection.
	emb = linear(scope.In("points_direct_project"), points) // [n, bs, 256]

	// Pooling (grid sample).
	x := Slice(points, AxisRange(), AxisRange(), AxisRange(0, 1)) // [n, bs, 1]
	y := Slice(points, AxisRange(), AxisRange(), AxisRange(1, 2))
	x = Squeeze(x, 2)
	y = Squeeze(y, 2)
	// grid_sample coords in [-1, 1].
	gridX := SubScalar(MulScalar(x, 2.0), 1.0)
	gridY := SubScalar(MulScalar(y, 2.0), 1.0)
	// Denormalize to pixel coords (align_corners=False).
	h, w := imgForPool.Shape().Dim(2), imgForPool.Shape().Dim(3)
	px := MulScalar(AddScalar(gridX, 1.0), float64(w-1)/2.0)
	py := MulScalar(AddScalar(gridY, 1.0), float64(h-1)/2.0)
	px = TransposeAllAxes(px, 1, 0) // [bs, n]
	py = TransposeAllAxes(py, 1, 0)
	sampled := bilinearSample(imgForPool, py, px) // [bs, 256, n]
	sampled = TransposeAllAxes(sampled, 2, 0, 1)  // [n, bs, 256]
	emb = Add(emb, linear(scope.In("points_pool_project"), sampled))

	// Positional encoding projection.
	enc := posEncodePoints(g, points, half)
	emb = Add(emb, linear(scope.In("points_pos_enc_project"), enc))

	// Label embedding.
	labelEmb := embedding(scope.At("label_embed"), labels, 2, dModel) // [n, bs, 256]
	return Add(emb, labelEmb)
}

// posEncodePoints returns the point positional encoding cat([posX, posY]).
func posEncodePoints(g *Graph, points *Node, half int) *Node {
	x := Squeeze(Slice(points, AxisRange(), AxisRange(), AxisRange(0, 1)), 2)
	y := Squeeze(Slice(points, AxisRange(), AxisRange(), AxisRange(1, 2)), 2)
	posX := posEncode1D(g, x, half, 10000.0)
	posY := posEncode1D(g, y, half, 10000.0)
	return Concatenate([]*Node{posX, posY}, -1) // [n, bs, 256]
}

// encodeBoxes builds the box prompt embeddings (direct projection + RoIAlign
// pooling + positional encoding, summed) plus the label embedding.
func encodeBoxes(scope *model.Scope, boxes, labels, imgForPool *Node, dModel, half int) *Node {
	g := boxes.Graph()
	bs := boxes.Shape().Dim(1)
	n := boxes.Shape().Dim(0)
	roiSize := 7

	var emb *Node
	emb = linear(scope.In("boxes_direct_project"), boxes) // [n, bs, 256]

	// RoIAlign pooling.
	xyxy := boxCxcywhToXyxy(boxes) // [n, bs, 4] (normalized)
	// Denormalize to pixel coords.
	scale := Const(g, [][][]float32{{{float32(imgForPool.Shape().Dim(3)), float32(imgForPool.Shape().Dim(2)), float32(imgForPool.Shape().Dim(3)), float32(imgForPool.Shape().Dim(2))}}})
	xyxy = Mul(xyxy, scale)                                                        // [n, bs, 4]
	roi := roiAlign(imgForPool, xyxy, roiSize)                                     // [n*bs, 256, 7, 7]
	roi = conv2d(scope.In("boxes_pool_project"), roi, dModel, roiSize, roiSize, 0) // [n*bs, 256, 1, 1]
	roi = Reshape(roi, n, bs, dModel)
	emb = Add(emb, roi)

	// Positional encoding projection: cat([posY, posX, h, w]) -> Linear(258, 256).
	cx := Squeeze(Slice(boxes, AxisRange(), AxisRange(), AxisRange(0, 1)), 2)
	cy := Squeeze(Slice(boxes, AxisRange(), AxisRange(), AxisRange(1, 2)), 2)
	bw := Squeeze(Slice(boxes, AxisRange(), AxisRange(), AxisRange(2, 3)), 2)
	bh := Squeeze(Slice(boxes, AxisRange(), AxisRange(), AxisRange(3, 4)), 2)
	posX := posEncode1D(g, cx, half, 10000.0)
	posY := posEncode1D(g, cy, half, 10000.0)
	enc := Concatenate([]*Node{posY, posX, ExpandAxes(bw, -1), ExpandAxes(bh, -1)}, -1) // [n, bs, 258]
	emb = Add(emb, linear(scope.In("boxes_pos_enc_project"), enc))

	labelEmb := embedding(scope.At("label_embed"), labels, 2, dModel)
	return Add(emb, labelEmb)
}

// roiAlign performs RoIAlign (aligned, adaptive sampling) on a channels-first
// image. boxes is [n, bs, 4] (xyxy, pixel coords); returns [n*bs, C, size, size].
func roiAlign(img, boxes *Node, size int) *Node {
	g := img.Graph()
	c := img.Shape().Dim(1)
	n := boxes.Shape().Dim(0)
	bs := boxes.Shape().Dim(1)

	boxes = TransposeAllAxes(boxes, 1, 0, 2) // [bs, n, 4]
	x0 := Slice(boxes, AxisRange(), AxisRange(), AxisRange(0, 1))
	y0 := Slice(boxes, AxisRange(), AxisRange(), AxisRange(1, 2))
	x1 := Slice(boxes, AxisRange(), AxisRange(), AxisRange(2, 3))
	y1 := Slice(boxes, AxisRange(), AxisRange(), AxisRange(3, 4))
	x0 = Squeeze(x0, 2) // [bs, n]
	y0 = Squeeze(y0, 2)
	x1 = Squeeze(x1, 2)
	y1 = Squeeze(y1, 2)

	roiW := Sub(x1, x0)
	roiH := Sub(y1, y0)
	binW := DivScalar(roiW, float64(size))
	binH := DivScalar(roiH, float64(size))

	// Output grid coordinates: j in [0,size) for x, i in [0,size) for y.
	j := IotaFull(g, shapes.Make(dtypes.Float32, 1, 1, 1, size))
	i := IotaFull(g, shapes.Make(dtypes.Float32, 1, 1, size, 1))
	// xs[bs, n, 1, size] = x0[bs,n,1,1] + (j + 0.5) * binW[bs,n,1,1]
	xs := Add(
		ExpandAxes(ExpandAxes(x0, 2), 3),
		Mul(AddScalar(j, 0.5), ExpandAxes(ExpandAxes(binW, 2), 3)),
	)
	// ys[bs, n, size, 1] = y0[bs,n,1,1] + (i + 0.5) * binH[bs,n,1,1]
	ys := Add(
		ExpandAxes(ExpandAxes(y0, 2), 2),
		Mul(AddScalar(i, 0.5), ExpandAxes(ExpandAxes(binH, 2), 2)),
	)

	xs = BroadcastToDims(xs, bs, n, size, size)
	ys = BroadcastToDims(ys, bs, n, size, size)

	// Sample at each (bs, n, size, size) point.
	out := bilinearSample(img, ys, xs)         // [bs, C, n, size, size]
	out = TransposeAllAxes(out, 0, 2, 1, 3, 4) // [bs, n, C, size, size]
	return Reshape(out, bs*n, c, size, size)
}

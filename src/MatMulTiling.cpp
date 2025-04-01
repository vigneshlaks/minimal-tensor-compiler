//===- MatMulTiling.cpp - Tiling pass for MatMul operations -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for tiling matrix multiplication operations
// that have been converted to linalg.generic operations.
//
//===----------------------------------------------------------------------===//

#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-tiling"

using namespace mlir;

namespace {

static bool isMatmulGenericOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  // Check inputs/outputs: 2 inputs and 1 output.
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check dimensions: matmul requires 3 loops (m, n, k).
  if (genericOp.getNumLoops() != 3)
    return false;

  // Check iterator types: [parallel, parallel, reduction].
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::reduction)
    return false;

  // Check indexing maps.
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (indexingMaps.size() != 3)
    return false;

  // Create affine expressions for checking the maps.
  AffineExpr m, n, k;
  bindDims(op->getContext(), m, n, k);

  // Expected maps for matmul:
  // LHS map: (m, n, k) -> (m, k)
  // RHS map: (m, n, k) -> (k, n)
  // Result map: (m, n, k) -> (m, n)
  auto expectedLhsMap = AffineMap::get(3, 0, {m, k}, op->getContext());
  auto expectedRhsMap = AffineMap::get(3, 0, {k, n}, op->getContext());
  auto expectedResultMap = AffineMap::get(3, 0, {m, n}, op->getContext());

  if (!(indexingMaps[0] == expectedLhsMap) &&
    indexingMaps[1] == expectedRhsMap &&
    indexingMaps[2] == expectedResultMap) {
    return false;
  }

  // Check the body to ensure it computes a*b + c.
  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  // Check for basic matmul operations: look for a multiply followed by an add.
  bool foundMul = false;
  bool foundAdd = false;
  bool foundYield = false;

  for (Operation &op : body) {
    if (isa<arith::MulFOp>(op))
      foundMul = true;
    else if (isa<arith::AddFOp>(op))
      foundAdd = true;
    else if (isa<linalg::YieldOp>(op))
      foundYield = true;
  }

  return foundMul && foundAdd && foundYield;
}

struct MatMulTilingPattern : public OpRewritePattern<linalg::GenericOp> {
  MatMulTilingPattern(MLIRContext *context, const linalg::LinalgTilingOptions &options)
      : OpRewritePattern<linalg::GenericOp>(context), options(options) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                               PatternRewriter &rewriter) const override {
    // Only apply to operations that represent matrix multiplication.
    if (!isMatmulGenericOp(op))
      return failure();

    // Check if this op has already been tiled.
    if (op->hasAttr("tiled"))
      return failure();

    // Apply tiling transformation.
    FailureOr<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(rewriter, op, options);
    if (failed(tiledOp))
      return failure();

    // Mark the newly created op as tiled to avoid re-tiling.
    Operation *tiledOperation = tiledOp->op;
    if (auto tiledGenericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(tiledOperation)) {
      tiledGenericOp -> setAttr("tiled", rewriter.getBoolAttr(true));
    }

    // Replace the original op with the tiled result.
    rewriter.replaceOp(op, tiledOp->tensorResults);

    return success();
  }

private:
  linalg::LinalgTilingOptions options;
};

struct MatMulTilingPass
    : public PassWrapper<MatMulTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTilingPass)

  MatMulTilingPass() = default;
  MatMulTilingPass(const MatMulTilingPass &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "matmul-tiling"; }

  StringRef getDescription() const final {
    return "Tile matrix multiplication operations for better performance";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Create tiling options.
    SmallVector<int64_t, 3> tileSizes = {tileSizeM, tileSizeN, tileSizeK};
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
    tilingOptions = tilingOptions.setLoopType(linalg::LinalgTilingLoopType::Loops);

    LLVM_DEBUG(llvm::dbgs() << "Applying MatMul tiling with sizes: ["
               << tileSizeM << ", " << tileSizeN << ", " << tileSizeK << "]\n");

    // Set up patterns.
    RewritePatternSet patterns(context);
    patterns.add<MatMulTilingPattern>(context, tilingOptions);

    // Apply patterns.
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  Option<int64_t> tileSizeM{
      *this, "tile-size-m", llvm::cl::desc("Tile size for M dimension (rows of first matrix)"),
      llvm::cl::init(64)};

  Option<int64_t> tileSizeN{
      *this, "tile-size-n", llvm::cl::desc("Tile size for N dimension (columns of second matrix)"),
      llvm::cl::init(64)};

  Option<int64_t> tileSizeK{
      *this, "tile-size-k", llvm::cl::desc("Tile size for K dimension (inner dimension)"),
      llvm::cl::init(64)};
};

} // namespace

namespace mlir {
namespace minimal {

std::unique_ptr<Pass> createMatMulTilingPass() {
  return std::make_unique<MatMulTilingPass>();
}

} // namespace minimal
} // namespace mlir
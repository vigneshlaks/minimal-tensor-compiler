//===- MatMulTiling.cpp - Tiling pass for MatMul operations -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for tiling matrix multiplication operations,
// specifically targeting linalg.matmul operations.
//
//===----------------------------------------------------------------------===//

#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {

// Define Tiling Patter
struct MatMulOpTilingPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatMulOpTilingPattern(MLIRContext *context, const linalg::LinalgTilingOptions &options)
      : OpRewritePattern<linalg::MatmulOp>(context), options(options) {}

  // Define how we rewrite operation
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                               PatternRewriter &rewriter) const override {

    if (op->hasAttr("tiled")) {
      return failure();
    }

    // Attempt to tile the operation
    FailureOr<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(rewriter, op, options);

    if (failed(tiledOp)) {
      return failure();
    }

    // Mark tile to avoid retiling attempt
    if (tiledOp->op) {
      tiledOp->op->setAttr("tiled", rewriter.getBoolAttr(true));
    }

    // Replace with tiled operation
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

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Find all matmul operations in the function
    funcOp.walk([&](linalg::MatmulOp matmulOp) {
      matmulOp->print(llvm::dbgs());
    });

    // Determine tile sizes
    SmallVector<int64_t, 3> tileSizes = {
      tileSizeM,  // M dimension tile size
      tileSizeN,  // N dimension tile size
      tileSizeK   // K dimension tile size
    };

    // Create tiling options
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);

    // Configure loop type
    if (useScfFor) {
      llvm::dbgs() << "Using SCF loops for tiling\n";
    } else {
      tilingOptions = tilingOptions.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);
      llvm::dbgs() << "Using parallel loops for tiling\n";
    }

    // Create rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<MatMulOpTilingPattern>(context, tilingOptions);

    // Apply the patterns
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Print completion
    llvm::dbgs() << "Completed MatMul Tiling Pass\n";
  }

  Option<int64_t> tileSizeM{
      *this, "tile-size-m", llvm::cl::desc("Tile size for M dimension (rows of first matrix)"),
      llvm::cl::init(32)};

  Option<int64_t> tileSizeN{
      *this, "tile-size-n", llvm::cl::desc("Tile size for N dimension (columns of second matrix)"),
      llvm::cl::init(32)};

  Option<int64_t> tileSizeK{
      *this, "tile-size-k", llvm::cl::desc("Tile size for K dimension (inner dimension)"),
      llvm::cl::init(32)};

  Option<bool> useScfFor{
      *this, "use-scf-for", llvm::cl::desc("Use scf.for instead of scf.parallel for tiling"),
      llvm::cl::init(true)};
};

} // namespace

namespace mlir {
namespace minimal {

std::unique_ptr<Pass> createMatMulTilingPass() {
  return std::make_unique<MatMulTilingPass>();
}

} // namespace minimal
} // namespace mlir
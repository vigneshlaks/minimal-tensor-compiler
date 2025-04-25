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
// This version uses TableGen-generated pass options.
//===----------------------------------------------------------------------===//

// Include the dialect header that includes TableGen declarations.
#include "MinimalDialect.h"

// Standard MLIR and LLVM includes.
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


namespace mlir {
namespace minimal {

#define GEN_PASS_DEF_MATMULTILING
#include "MinimalPasses.h.inc"

// Tiling pattern that rewrites linalg.matmul into tiled form.
struct MatMulOpTilingPattern : public OpRewritePattern<linalg::MatmulOp> {
  MatMulOpTilingPattern(MLIRContext *context, const linalg::LinalgTilingOptions &options)
      : OpRewritePattern<linalg::MatmulOp>(context), options(options) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    if (op->hasAttr("tiled"))
      return failure();

    // Tile the operation according to the provided tiling options.
    FailureOr<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(rewriter, op, options);
    if (failed(tiledOp))
      return failure();

    // Mark the tiled op so it is not retiled.
    if (tiledOp->op)
      tiledOp->op->setAttr("tiled", rewriter.getBoolAttr(true));

    // Replace the original op with the tiled op's tensor results.
    rewriter.replaceOp(op, tiledOp->tensorResults);
    return success();
  }

private:
  linalg::LinalgTilingOptions options;
};


struct MatMulTilingPass : public impl::MatMulTilingBase<MatMulTilingPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTilingPass)

  // Default constructors are provided by the base class.
  MatMulTilingPass() = default;
  MatMulTilingPass(const MatMulTilingPass &pass) : impl::MatMulTilingBase<MatMulTilingPass>(pass) {}

  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Use the TableGen-generated options for tiling.
    // tileSizeM, tileSizeN, and tileSizeK are available as member variables inherited from the base.
    auto opts = MatMulTilingOptions{};
    SmallVector<int64_t, 3> tileSizes{ opts.tileSizeM, opts.tileSizeN, opts.tileSizeK };

    // Configure tiling options based on command-line (or default) values.
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);

    // Create a pattern set and add the tiling pattern.
    RewritePatternSet patterns(context);
    patterns.add<MatMulOpTilingPattern>(context, tilingOptions);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    llvm::dbgs() << "Completed MatMul Tiling Pass\n";
  }
};

} // end anonymous namespace
}

namespace mlir {
namespace minimal {

std::unique_ptr<Pass> createMatMulTilingPass() {
  return std::make_unique<minimal::MatMulTilingPass>();
}

} // namespace minimal
} // namespace mlir
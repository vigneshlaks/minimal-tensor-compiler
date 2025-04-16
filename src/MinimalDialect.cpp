//===- MinimalDialect.cpp - Minimal dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinimalDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"


using namespace mlir;
using namespace mlir::minimal;

#include "MinimalDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Minimal dialect.
//===----------------------------------------------------------------------===//


void MinimalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MinimalOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// Minimal ops
//===----------------------------------------------------------------------===//


#define GET_OP_CLASSES
#include "MinimalOps.cpp.inc"


namespace mlir::minimal {

// Import passes
#define GEN_PASS_DEF_MATMULTOLINALG
#include "MinimalPasses.h.inc"
#define GEN_PASS_DEF_RELUFUSION
#include "MinimalPasses.h.inc"

//===----------------------------------------------------------------------===//
// Minimal passes
//===----------------------------------------------------------------------===//

namespace {

struct MatMulToLinalgPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // Get the result type using mlir::cast.
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    // Create a zero-filled tensor for output.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value filledTensor = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    // Create the linalg.matmul operation with properly separated inputs and outputs
    Value newOp = rewriter.create<linalg::MatmulOp>(
        loc,
        resultType,
        ValueRange{lhs, rhs},
        ValueRange{filledTensor}
    ).getResult(0);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

  struct MatMulToLinalg : public impl::MatMulToLinalgBase<MatMulToLinalg> {
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
    }

    void runOnOperation() override {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      patterns.add<MatMulToLinalgPattern>(context);

      // Set up a conversion target that marks MatMulOp as illegal.
      ConversionTarget target(*context);
      target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect, func::FuncDialect>();
      target.addIllegalOp<MatMulOp>();

      if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }
  };
}

namespace {
struct MatMulReluFusionPattern : public OpRewritePattern<minimal::ReluOp> {
  using OpRewritePattern<minimal::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(minimal::ReluOp reluOp,
                               PatternRewriter &rewriter) const override {
    // Check if the input to ReLU is a MatMul operation
    auto matmulOp = reluOp.getInput().getDefiningOp<minimal::MatMulOp>();
    if (!matmulOp)
      return failure();

    // Check if the MatMul has other uses
    if (!matmulOp.getResult().hasOneUse())
      return failure();

    // Create a fused operation at the location of the MatMul
    Location loc = matmulOp.getLoc();
    Value fusedOp = rewriter.create<minimal::FusedMatMulReluOp>(
        loc,
        reluOp.getType(),
        matmulOp.getLhs(),
        matmulOp.getRhs());

    // Replace ReLU with the fused operation result
    rewriter.replaceOp(reluOp, fusedOp);

    // The original MatMul is now dead as its only use was replaced
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

struct ReluFusion : public impl::ReluFusionBase<ReluFusion> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<minimal::MinimalDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MatMulReluFusionPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
}

std::unique_ptr<mlir::Pass> createReluFusionPass() {
  return std::make_unique<ReluFusion>();
}


std::unique_ptr<mlir::Pass> createMatMulToLinalgPass() {
  return std::make_unique<MatMulToLinalg>();
}

} // namespace mlir::minimal


//===----------------------------------------------------------------------===//
// Minimal types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "MinimalTypes.cpp.inc"

void MinimalDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MinimalTypes.cpp.inc"
      >();
}

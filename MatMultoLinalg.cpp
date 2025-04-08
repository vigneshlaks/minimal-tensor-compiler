//===- MatMulToLinalg.cpp - Convert MatMul to Linalg ops -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace minimal {

// Tablegen
#define GEN_PASS_DEF_MATMULTOLINALG
#include "include/MinimalPasses.h.inc"

namespace {
// Rewrite pattern
struct MatMulToLinalgPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  //Specify how to rewrite
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

    // Create the linalg.matmul operation.
    // Note: Pack operands into a ValueRange.
    Value newOp = rewriter.create<linalg::MatmulOp>(
        loc, resultType, ValueRange({lhs, rhs, filledTensor})
    ).getResult(0);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};


struct MatMulToLinalg : public impl::MatMulToLinalgBase<MatMulToLinalg> {
  // Leaving getDependentDialects for now
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
} // end anonymous namespace

// Pass registration.
std::unique_ptr<Pass> createMatMulToLinalgPass() {
  return std::make_unique<MatMulToLinalg>();
}

} // namespace minimal
} // namespace mlir
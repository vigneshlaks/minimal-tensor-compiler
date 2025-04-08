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
#define GEN_PASS_DEF_MINIMALSWITCHBARFOO
#define GEN_PASS_DEF_MATMULTOLINALG
#include "MinimalPasses.h.inc"

//===----------------------------------------------------------------------===//
// Minimal passes
//===----------------------------------------------------------------------===//

namespace {
class MinimalSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class MinimalSwitchBarFoo
    : public impl::MinimalSwitchBarFooBase<MinimalSwitchBarFoo> {
public:
  using impl::MinimalSwitchBarFooBase<
      MinimalSwitchBarFoo>::MinimalSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MinimalSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


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
        ValueRange{lhs, rhs},    // inputs (A and B matrices)
        ValueRange{filledTensor} // output (C matrix)
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
  };;




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

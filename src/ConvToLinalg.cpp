//===- ConvToLinalg.cpp - Convert Conv to Linalg ops ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinimalDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace minimal {
#define GEN_PASS_DEF_CONVTOLINALG
#include "MinimalPasses.h.inc"

namespace {
struct ConvToLinalgPattern : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
                              PatternRewriter &rewriter) const override {
    // Convert minimal.conv to linalg.conv_2d
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value filter = op.getFilter();
    
    // Create a zero-filled output tensor
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto outputElemType = outputType.getElementType();
    
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(outputElemType));
    
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputElemType);
    
    Value filledTensor = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);
    
    // Create the linalg convolution operation
    Value result = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        loc,
        TypeRange{outputType},
        ValueRange{input, filter},
        ValueRange{filledTensor},
        ArrayRef<NamedAttribute>{}
    ).getResult(0);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Define the actual pass
struct ConvToLinalg : public impl::ConvToLinalgBase<ConvToLinalg> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvToLinalgPattern>(context);
    
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<ConvOp>();
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
}

// Pass registration
std::unique_ptr<Pass> createConvToLinalgPass() {
  return std::make_unique<ConvToLinalg>();
}

} // namespace minimal
} // namespace mlir
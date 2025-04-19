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
//===----------------------------------------------------------------------===//
// Minimal passes
//===----------------------------------------------------------------------===//

namespace {

  

  struct MatMulToLinalgPattern : public OpRewritePattern<MatMulOp> {
    using OpRewritePattern<MatMulOp>::OpRewritePattern;
  
    LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      Value lhs = op.getLhs();
      Value rhs = op.getRhs();
  
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

  struct ReluToLinalgPattern : public OpRewritePattern<ReluOp> {
    using OpRewritePattern<ReluOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ReluOp op, PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      Value input = op.getInput();

      auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(resultType.getElementType()));

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType());
        
      SmallVector<AffineMap> indexingMaps = {
        AffineMap::getMultiDimIdentityMap(resultType.getRank(), rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(resultType.getRank(), rewriter.getContext())
      };

      SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(), 
                                                   utils::IteratorType::parallel);
      

      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc,
          TypeRange{resultType},
          ValueRange{input},
          ValueRange{emptyTensor},
          indexingMaps,
          iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {

            Value inputVal = args[0];
            Value maxOp = nestedBuilder.create<arith::MaximumFOp>(nestedLoc, inputVal, zero);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, maxOp);
          }
      );
      
      rewriter.replaceOp(op, genericOp.getResult(0));
      return success();        
    }
  };

  #define GEN_PASS_DEF_NNLOWERING
  #include "MinimalPasses.h.inc"
  
  struct NNToLinalgPass : public impl::NNLoweringBase<NNToLinalgPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
    }

    void runOnOperation() override {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.add<MatMulToLinalgPattern, ReluToLinalgPattern>(context);

      ConversionTarget target(*context);
      target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect, func::FuncDialect>();
      target.addIllegalOp<MatMulOp, ReluOp>();

      if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }
  };

}

std::unique_ptr<mlir::Pass> createNNToLinalgPass() {
  return std::make_unique<NNToLinalgPass>();
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

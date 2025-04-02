//===- MatMulToLinalg.cpp - Conversion from Minimal to Linalg dialect ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct MatMulToLinalgPattern : public OpRewritePattern<minimal::MatMulOp> {
  using OpRewritePattern<minimal::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(minimal::MatMulOp op,
                               PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get types
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());

    if (!lhsType || !rhsType || !resultType)
      return failure();

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || resultType.getRank() != 2) {
      return op.emitError("MatMul requires 2D tensors");
    }

    if (lhsType.getDimSize(1) != rhsType.getDimSize(0) ||
        lhsType.getDimSize(0) != resultType.getDimSize(0) ||
        rhsType.getDimSize(1) != resultType.getDimSize(1)) {
        return op.emitError("MatMul dimensions are inconsistent");
    }

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));

    // Fill output tensor with zeros
    Value zeroTensor = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);

    auto lhsMap = AffineMap::get(3, 0, {m, k}, rewriter.getContext());
    auto rhsMap = AffineMap::get(3, 0, {k, n}, rewriter.getContext());
    auto resultMap = AffineMap::get(3, 0, {m, n}, rewriter.getContext());

    // Create vector of affine maps and iterator types
    SmallVector<AffineMap, 3> indexingMaps = {lhsMap, rhsMap, resultMap};

    SmallVector<utils::IteratorType, 3> iteratorTypes = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction
    };

    // Create linalg.generic for matmul
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{resultType},
        ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{zeroTensor},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value lhs = args[0];
          Value rhs = args[1];
          Value acc = args[2];

          Value mul = nestedBuilder.create<arith::MulFOp>(nestedLoc, lhs, rhs);

          Value add = nestedBuilder.create<arith::AddFOp>(nestedLoc, acc, mul);

          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });


    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MatMulToLinalgPass
    : public PassWrapper<MatMulToLinalgPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulToLinalgPass)

    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                      arith::ArithDialect>();
    }

    StringRef getArgument() const final { return "minimal-matmul-to-linalg"; }

    void runOnOperation() override {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.add<MatMulToLinalgPattern>(context);


      ConversionTarget target(*context);
      target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                             arith::ArithDialect, func::FuncDialect>();
      target.addIllegalOp<minimal::MatMulOp>();

      if (failed(applyPartialConversion(getOperation(), target,
                                       std::move(patterns)))) {
        signalPassFailure();
      }
    }
  };
}

std::unique_ptr<mlir::Pass> mlir::minimal::createMatMulToLinalgPass() {
  return std::make_unique<MatMulToLinalgPass>();
}
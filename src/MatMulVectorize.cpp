//===- MatMulVectorize.cpp - Vectorization pass for tiled MatMul ops -----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for vectorizing matrix multiplication operations
// that have already been tiled.
//
//===----------------------------------------------------------------------===//

#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

#define DEBUG_TYPE "matmul-vectorization"

using namespace mlir;

namespace {

static bool isTiledMatmulGenericOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  if (!genericOp->hasAttr("tiled"))
    return false;

  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
    return false;

  if (genericOp.getNumLoops() != 3)
    return false;

  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::reduction)
    return false;

  return true;
}

struct MatMulVectorizationPattern : public OpRewritePattern<linalg::GenericOp> {
  MatMulVectorizationPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                               PatternRewriter &rewriter) const override {
    if (!isTiledMatmulGenericOp(op))
      return failure();

    if (op->hasAttr("vectorized"))
      return failure();

    Block &body = op.getRegion().front();

    if (body.getOperations().size() < 3)
      return failure();

    Operation *mulOp = nullptr;
    Operation *addOp = nullptr;
    for (Operation &bodyOp : body.getOperations()) {
      if (isa<arith::MulFOp>(bodyOp)) {
        mulOp = &bodyOp;
        llvm::errs() << "Found multiplication op: " << *mulOp << "\n";

        llvm::errs() << "  Operands:\n";
        for (Value operand : mulOp->getOperands()) {
          llvm::errs() << "    " << operand << " : " << operand.getType() << "\n";
        }
      }
      else if (isa<arith::AddFOp>(bodyOp))
        addOp = &bodyOp;
    }

    if (!mulOp || !addOp)
      return failure();

    Location loc = op.getLoc();

    auto indexingMaps = op.getIndexingMapsArray();
    auto outputMap = indexingMaps.back();
    auto resultType = mlir::cast<MemRefType>(op.getOutputs()[0].getType());

    SmallVector<int64_t, 4> shape;
    for (auto dim : resultType.getShape())
      shape.push_back(dim);

    if (shape.size() < 2)
      return failure();

    int64_t mValue = shape[shape.size() - 2];
    int64_t nValue = shape[shape.size() - 1];

    auto inputType = mlir::cast<MemRefType>(op.getInputs()[0].getType());
    SmallVector<int64_t, 4> inputShape;
    for (auto dim : inputType.getShape())
      inputShape.push_back(dim);

    if (inputShape.size() < 2)
      return failure();

    int64_t kValue = inputShape[inputShape.size() - 1];

    Type elementType = resultType.getElementType();

    VectorType aVectorType = VectorType::get({mValue, kValue}, elementType);
    VectorType bVectorType = VectorType::get({kValue, nValue}, elementType);
    VectorType cVectorType = VectorType::get({mValue, nValue}, elementType);

    op->setAttr("a_vector_type", TypeAttr::get(aVectorType));
    op->setAttr("b_vector_type", TypeAttr::get(bVectorType));
    op->setAttr("c_vector_type", TypeAttr::get(cVectorType));

    op->setAttr("vectorized", rewriter.getBoolAttr(true));

    return success();
  }
};

struct MatMulVectorizationPass
    : public PassWrapper<MatMulVectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationPass)

  MatMulVectorizationPass() = default;
  MatMulVectorizationPass(const MatMulVectorizationPass &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
  }

  StringRef getArgument() const final { return "matmul-vectorize"; }

  StringRef getDescription() const final {
    return "Vectorize tiled matrix multiplication operations for better performance";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "Applying MatMul vectorization with vector size: "
               << vectorSize << "\n");

    {
      RewritePatternSet patterns(context);
      patterns.add<MatMulVectorizationPattern>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }

  Option<int64_t> vectorSize{
      *this, "vector-size", llvm::cl::desc("Vector size for SIMD operations"),
      llvm::cl::init(8)};
};

} // namespace

namespace mlir {
namespace minimal {

std::unique_ptr<Pass> createMatMulVectorizationPass() {
  return std::make_unique<MatMulVectorizationPass>();
}

} // namespace minimal
} // namespace mlir
//===- MatMulBufferize.cpp - Bufferization pass for MatMul operations ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for bufferizing matrix multiplication operations
// that have been tiled.
//
//===----------------------------------------------------------------------===//

#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-bufferize"

using namespace mlir;

namespace {

// Helper to check if an operation is a tiled matrix multiplication
static bool isTiledMatmulGenericOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  // Check if it's marked as tiled
  if (!genericOp->hasAttr("tiled"))
    return false;

  // Check inputs/outputs: 2 inputs and 1 output
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check iterator types: [parallel, parallel, reduction]
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::reduction)
    return false;

  return true;
}

// Pattern to convert tensor.extract_slice to memref.subview
struct ExtractSliceToSubViewPattern : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                               PatternRewriter &rewriter) const override {
    Value source = op.getSource();

    // Check if the source is a tensor type
    auto tensorType = source.getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return failure();

    // Create a memref type with the same shape and element type
    auto memrefType = MemRefType::get(tensorType.getShape(),
                                     tensorType.getElementType());

    // Create ToMemrefOp to convert tensor to memref
    Value memref = rewriter.create<bufferization::ToMemrefOp>(
        op.getLoc(), memrefType, source);

    // Create a subview operation
    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(),
        memref,
        op.getMixedOffsets(),
        op.getMixedSizes(),
        op.getMixedStrides());

    // Create tensor from memref for compatibility with existing IR
    auto resultType = op.getResult().getType().cast<RankedTensorType>();
    auto fromMemref = rewriter.create<bufferization::ToTensorOp>(
        op.getLoc(), resultType, subview);

    rewriter.replaceOp(op, fromMemref);
    return success();
  }
};

// Pattern to convert tensor.insert_slice to memref-based operations
struct InsertSliceToMemRefPattern : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                               PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Value dest = op.getDest();

    // Check if source and dest are tensor types
    auto sourceTensorType = source.getType().dyn_cast<RankedTensorType>();
    auto destTensorType = dest.getType().dyn_cast<RankedTensorType>();
    if (!sourceTensorType || !destTensorType)
      return failure();

    // Create memref types
    auto sourceMemRefType = MemRefType::get(
        sourceTensorType.getShape(),
        sourceTensorType.getElementType());

    auto destMemRefType = MemRefType::get(
        destTensorType.getShape(),
        destTensorType.getElementType());

    // Convert tensors to memrefs
    Value sourceMemref = rewriter.create<bufferization::ToMemrefOp>(
        op.getLoc(), sourceMemRefType, source);

    Value destMemref = rewriter.create<bufferization::ToMemrefOp>(
        op.getLoc(), destMemRefType, dest);

    // Create a subview of the destination
    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(),
        destMemref,
        op.getMixedOffsets(),
        op.getMixedSizes(),
        op.getMixedStrides());

    // Copy from source to the subview
    rewriter.create<memref::CopyOp>(op.getLoc(), sourceMemref, subview);

    // Create a tensor from the modified memref for compatibility
    auto resultType = op.getResult().getType().cast<RankedTensorType>();
    auto result = rewriter.create<bufferization::ToTensorOp>(
        op.getLoc(), resultType, destMemref);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Pattern to convert linalg.generic on tensors to linalg.generic on memrefs
struct GenericTensorToMemRefPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                               PatternRewriter &rewriter) const override {
    // Only match tiled matmul operations
    if (!isTiledMatmulGenericOp(op))
      return failure();

    Location loc = op.getLoc();

    // Check if all operands are tensor types and convert them
    SmallVector<Value, 4> memrefInputs;
    for (Value input : op.getDpsInputs()) {
      auto tensorType = input.getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        return failure();

      auto memrefType = MemRefType::get(
          tensorType.getShape(),
          tensorType.getElementType());

      Value memref = rewriter.create<bufferization::ToMemrefOp>(
          loc, memrefType, input);
      memrefInputs.push_back(memref);
    }

    SmallVector<Value, 2> memrefOutputs;
    for (Value output : op.getDpsInits()) {
      auto tensorType = output.getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        return failure();

      auto memrefType = MemRefType::get(
          tensorType.getShape(),
          tensorType.getElementType());

      Value memref = rewriter.create<bufferization::ToMemrefOp>(
          loc, memrefType, output);
      memrefOutputs.push_back(memref);
    }

    // Create new generic op on memrefs
    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{},  // No tensor results for memref version
        /*inputs=*/memrefInputs,
        /*outputs=*/memrefOutputs,
        op.getIndexingMapsArray(),
        op.getIteratorTypesArray());

    // Clone the body from the original op
    rewriter.cloneRegionBefore(op.getRegion(), newGeneric.getRegion(), newGeneric.getRegion().begin());

    // Create tensor results from memref outputs for compatibility
    SmallVector<Value, 2> results;
    for (unsigned i = 0; i < memrefOutputs.size(); ++i) {
      Value output = memrefOutputs[i];
      auto memrefType = output.getType().cast<MemRefType>();

      // Get the original result type
      auto resultTensorType = op->getResult(i).getType().cast<RankedTensorType>();

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, resultTensorType, output);
      results.push_back(tensor);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MatMulBufferizePass
    : public PassWrapper<MatMulBufferizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulBufferizePass)

  MatMulBufferizePass() = default;
  MatMulBufferizePass(const MatMulBufferizePass &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "matmul-bufferize"; }

  StringRef getDescription() const final {
    return "Bufferize matrix multiplication operations by converting tensor operations to memref operations";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "Applying MatMul bufferization\n");

    // Set up patterns
    RewritePatternSet patterns(context);
    patterns.add<ExtractSliceToSubViewPattern>(context);
    patterns.add<InsertSliceToMemRefPattern>(context);
    patterns.add<GenericTensorToMemRefPattern>(context);

    // Apply patterns
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace minimal {

std::unique_ptr<Pass> createMatMulBufferizePass() {
  return std::make_unique<MatMulBufferizePass>();
}

} // namespace minimal
} // namespace mlir
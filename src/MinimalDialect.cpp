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
  namespace {
    #define GEN_PASS_DEF_MATMULRELUFUSION
    #include "MinimalPasses.h.inc"


    class FuseMatmulReLUPattern : public OpRewritePattern<linalg::GenericOp> {
    public:
      using OpRewritePattern::OpRewritePattern;
    
      LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                    PatternRewriter &rewriter) const override {
        // Check for one input and one output
        if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
          return failure();

        // Check the block
        Region &region = genericOp.getRegion();
        if (!llvm::hasSingleElement(region))
          return failure();

        // Get the single block and its terminator
        Block &block = region.front();
        auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
        if (!yieldOp)
          return failure();
        auto maxOp = dyn_cast<arith::MaximumFOp>(yieldOp.getOperand(0)
                                                    .getDefiningOp());
        if (!maxOp)
          return failure();
    
        // Check for matmul producer
        auto matmulOp = genericOp.getDpsInputOperand(0)->get()
                            .getDefiningOp<linalg::MatmulOp>();
        if (!matmulOp)
          return failure();

        // Build fused generic
        Location loc = genericOp.getLoc();
        
        // Get the indexing map and iterator types from the matmul
        SmallVector<AffineMap> maps = matmulOp.getIndexingMapsArray();
        SmallVector<utils::IteratorType> itTypes = matmulOp.getIteratorTypesArray();
    
        // Element type for the constant zero we'll re‑create inside the fused region
        auto tensorType = genericOp.getResultTypes()[0]
                              .cast<RankedTensorType>();
        Type eltType = tensorType.getElementType();
    
        // Inputs: the original two matmul operands
        Value lhs = matmulOp.getDpsInputOperand(0)->get();
        Value rhs = matmulOp.getDpsInputOperand(1)->get();

        // Init/output: the generic's init-tensor (same shape as its result)
        Value init = genericOp.getDpsInitOperand(0)->get();
    
        // Create the new fused generic
        auto fused = rewriter.create<linalg::GenericOp>(
            loc,
            genericOp.getResultTypes(),
            ValueRange{lhs, rhs},
            ValueRange{init},
            maps,
            itTypes,
            [&](OpBuilder &nestedBuilder,
                Location nestedLoc,
                ValueRange args) {
              // args = [A(i,j), B(j,k), Acc(i,k)]
              Value a = args[0];
              Value b = args[1];
              Value acc = args[2];
              // dot-accumulate step
              Value prod = nestedBuilder.create<arith::MulFOp>(nestedLoc, a, b);
              Value sum = nestedBuilder.create<arith::AddFOp>(
                  nestedLoc, prod, acc);
              // Relu max(sum, 0)
              Value zero = nestedBuilder.create<arith::ConstantOp>(
                  nestedLoc,
                  nestedBuilder.getZeroAttr(eltType));
              Value relu = nestedBuilder.create<arith::MaximumFOp>(
                  nestedLoc, sum, zero);
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, relu);
            });
    
        // Replace with fused result and remove the matmul
        rewriter.replaceOp(genericOp, fused.getResults());
        rewriter.eraseOp(matmulOp);
        return success();
      }
    };
    
    // This is the actual pass that will run the fusion pattern
    struct FuseMatmulReLUPass : public impl::MatMulReluFusionBase<FuseMatmulReLUPass> {
      void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect, 
                       mlir::tensor::TensorDialect>();
      }
    
      void runOnOperation() override {
        mlir::MLIRContext *context = &getContext();
        mlir::RewritePatternSet patterns(context);
        
        // Add our custom pattern
        patterns.add<FuseMatmulReLUPattern>(context);
        
        // potentially useful patten from Linalg
        mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
        
        // Apply patterns
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
          signalPassFailure();
        }
      }
    };
    
    } // end anonymous namespace
}

std::unique_ptr<mlir::Pass> createNNToLinalgPass() {
  return std::make_unique<NNToLinalgPass>();
}

std::unique_ptr<mlir::Pass> createFusedMatmulReLUPass() {
  return std::make_unique<FuseMatmulReLUPass>();
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

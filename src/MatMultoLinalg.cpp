#include "../include/MinimalDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct MinimalMatMulToLinalgMatMulPattern : public RewritePattern {
  MinimalMatMulToLinalgMatMulPattern(MLIRContext *context)
      : RewritePattern(mlir::minimal::MatMulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = cast<mlir::minimal::MatMulOp>(op);
    auto loc = op->getLoc();

    Value lhs = matmulOp.getLhs();
    Value rhs = matmulOp.getRhs();
    Value result = matmulOp.getResult();

    auto resultType = mlir::cast<RankedTensorType>(result.getType());
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value zeroTensor = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{emptyTensor}).getResult(0);

    Value linalgMatmul = rewriter.create<linalg::MatmulOp>(
        loc,
        TypeRange{resultType},
        ValueRange{lhs, rhs},
        ValueRange{zeroTensor}).getResult(0);

    rewriter.replaceOp(matmulOp, linalgMatmul);

    return success();
  }
};

// The actual pass that uses the pattern above
struct MatMulToLinalgPass
    : public PassWrapper<MatMulToLinalgPass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "minimal-matmul-to-linalg"; }

  StringRef getDescription() const final {
    return "Convert Minimal MatMul operations to Linalg implementation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<MinimalMatMulToLinalgMatMulPattern>(patterns.getContext());

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<mlir::minimal::MatMulOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};



} // end anonymous namespace

namespace mlir {

namespace minimal {

  std::unique_ptr<Pass> mlir::minimal::createMatMulToLinalgPass() {
    return std::make_unique<MatMulToLinalgPass>();

  }

}

}
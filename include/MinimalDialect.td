//===- MinimalDialect.td - Minimal dialect ----      -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MINIMAL_DIALECT
#define MINIMAL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Minimal dialect definition.
//===----------------------------------------------------------------------===//

def Minimal_Dialect : Dialect {
    let name = "minimal";
    let summary = "A minimal out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::minimal";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Operation definition.
//===----------------------------------------------------------------------===//

class Minimal_Op<string mnemonic, list<Trait> traits = []> :
        Op<Minimal_Dialect, mnemonic, traits>;

def MatMulOp : Minimal_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication C = A * B where A is MxK and B is KxN.

    Example:
    ```mlir
    %C = minimal.matmul %A, %B : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
    ```
  }];

  let arguments = (ins
    TensorOf<[AnyType]>:$lhs,
    TensorOf<[AnyType]>:$rhs
  );

  let results = (outs
    TensorOf<[AnyType]>:$output
  );

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def Minimal_ReluOp : Minimal_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies ReLU (max(0, x)) to each element of the input tensor.

    Example:
    ```mlir
    %b = minimal.relu %a : tensor<1x128xf32> -> tensor<1x128xf32>
    ```
  }];

  let arguments = (ins
    TensorOf<[AnyFloat]>:$input
  );

  let results = (outs
    TensorOf<[AnyFloat]>:$output
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Pass definition.
//===----------------------------------------------------------------------===//

def NNLowering: Pass<"nn-lowering", "::mlir::func::FuncOp"> {
  let summary = "Lower minimal relu and matmul to Linalg";
  let description = [{
    This pass lowers neural network operations to linalg.generic operations.
  }];
  let constructor = "mlir::minimal::createNNToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

// MatMulTiling pass
def MatMulTiling: Pass<"linalg-tiling", "::mlir::func::FuncOp"> {
  let summary = "Tile matrix multiplication operations";
  let description = [{
    This pass tiles matrix multiplication operations that have been lowered to
    linalg.generic operations with specific access patterns.
  }];
  let constructor = "mlir::minimal::createMatMulTilingPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];

  let options = [
    Option<"tileSizeM", "tile-size-m", "int64_t", "64",
           "Tile size for M dimension (rows of first matrix)">,
    Option<"tileSizeN", "tile-size-n", "int64_t", "64",
           "Tile size for N dimension (columns of second matrix)">,
    Option<"tileSizeK", "tile-size-k", "int64_t", "64",
           "Tile size for K dimension (inner dimension)">
  ];
}


def MatMulVectorize: Pass<"matmul-vectorize", "::mlir::func::FuncOp"> {
   let summary = "Vectorize matrix multiplication operation";

   let description = [{
       Transforms tiled matrix multiplication operations into vectorized form.
       This pass identifies linalg.generic operations that represent matrix
       multiplications and have already been tiled, then transforms them to
       use vector operations for improved performance on SIMD hardware.
   }];

   let constructor = "mlir::minimal::createMatMulVectorizationPass()";
   let dependentDialects = [
       "mlir::linalg::LinalgDialect",
       "mlir::vector::VectorDialect",
       "mlir::scf::SCFDialect",
       "mlir::tensor::TensorDialect",
       "mlir::arith::ArithDialect"
   ];

   let options = [
       Option<"vectorSize", "vector-size", "int64_t", "8",
              "Vector size for SIMD operations">,
       Option<"lowerVectorContracts", "lower-vector-contracts", "bool", "true",
              "Lower vector.contract operations to vector.fma">
   ];
}


// MatMulBufferize pass
def MatMulBufferize: Pass<"matmul-bufferize", "::mlir::func::FuncOp"> {
    let summary = "Bufferize matrix multiplication operations";
    let description = [{
    This pass transforms tensor-based matrix multiplication operations that have
    been tiled to use memory buffers (memrefs) instead of tensors. Bufferization
    is a key step in lowering high-level tensor operations to lower-level code
    that operates on memory buffers.

    The pass specifically looks for tiled linalg.generic operations with the "tiled"
    attribute and transforms:
    - tensor.extract_slice to memref.subview
    - tensor.insert_slice to memref-based operations
    - linalg.generic on tensors to linalg.generic on memrefs
    }];
    let constructor = "mlir::minimal::createMatMulBufferizePass()";
    let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
    ];
}

//===----------------------------------------------------------------------===//
// Type definitions.
//===----------------------------------------------------------------------===//

class Minimal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Minimal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Minimal_CustomType : Minimal_Type<"Custom", "custom"> {
    let summary = "Minimal custom type";
    let description = "Custom type in minimal dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}


#endif // MINIMAL_DIALECT

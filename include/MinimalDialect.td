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
// Base minimal operation definition.
//===----------------------------------------------------------------------===//

class Minimal_Op<string mnemonic, list<Trait> traits = []> :
        Op<Minimal_Dialect, mnemonic, traits>;

def Minimal_FooOp : Minimal_Op<"foo", [Pure, SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `minimal.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = minimal.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def ConvOp : Minimal_Op<"conv", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Performs a 2D convolution with the given input and filter tensors.
  }];
  
  let arguments = (ins
    TensorOf<[AnyType]>:$input,
    TensorOf<[AnyType]>:$filter
  );
  
  let results = (outs
    TensorOf<[AnyType]>:$output
  );
  
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($output)
  }];
}

def MinimalSwitchBarFoo: Pass<"minimal-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def ConvToLinalg: Pass<"minimal-conv-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Convert Minimal Conv operations to Linalg operations";
  let description = [{
    This pass lowers minimal.conv operations to linalg.conv_2d_nhwc_hwcf operations.
    
    Example:
    ```mlir
    %conv = minimal.conv %input, %filter : tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32> -> tensor<1x26x26x32xf32>
    ```
    
    Gets converted to a linalg.conv_2d_nhwc_hwcf operation with appropriate tensor setup.
  }];
  let constructor = "mlir::minimal::createConvToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

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

// Matrix Multiplication Operation
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

// MatMulToLinalg pass
def MatMulToLinalg: Pass<"minimal-matmul-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Convert Minimal MatMul operations to Linalg operations";
  let description = [{
    This pass lowers minimal.matmul operations to linalg.generic operations
    representing matrix multiplication.

    Example:
    ```mlir
    %C = minimal.matmul %A, %B : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
    ```

    Gets converted to a linalg.generic operation with appropriate tensor setup.
  }];
  let constructor = "mlir::minimal::createMatMulToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

// MatMulTiling pass
def MatMulTiling: Pass<"matmul-tiling", "::mlir::func::FuncOp"> {
  let summary = "Tile matrix multiplication operations";
  let description = [{
    This pass tiles matrix multiplication operations that have been lowered to
    linalg.generic operations with specific access patterns. Tiling improves
    cache locality and enables further optimizations like vectorization and parallelization.
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


#endif // MINIMAL_DIALECT

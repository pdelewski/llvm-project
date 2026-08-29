//===- IRMutationObserver.h - Structural IR mutation hooks -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EXPERIMENTAL (mlir-obs local patch, phase 1).
//
// Rewriter listeners only observe mutations that go through a RewriterBase.
// A pass that calls Operation::moveBefore, Block::getOperations().splice, or
// creates/erases ops directly is invisible to them. This observer sits under
// all of that, on the ilist trait methods every op-list mutation funnels
// through, so it sees attach/detach/move regardless of which API performed
// the mutation.
//
// Notify-only contract: implementations must not mutate IR, must not remove
// themselves during a callback, and must be internally synchronized — the
// hooks fire on whichever thread performs the mutation (the pass manager is
// multi-threaded).
//
// The registration is deliberately process-global for this experiment: the
// hot-path guard is a single relaxed atomic load, so compiles without an
// observer pay one predictable branch per list mutation. The upstream-shaped
// version of this would live on MLIRContext; that migration is mechanical
// and intentionally deferred until the overhead numbers justify the RFC.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_IRMUTATIONOBSERVER_H
#define MLIR_IR_IRMUTATIONOBSERVER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

#include <atomic>

namespace mlir {
class Block;
class Operation;

class IRMutationObserver {
public:
  virtual ~IRMutationObserver() = default;

  /// `op` was linked into a block's operation list (creation, parsing, or
  /// re-insertion after a detach).
  virtual void notifyOperationAttached(Operation *op) {}

  /// `op` was unlinked from its block's operation list. The op is still
  /// alive here; destruction, if any, happens after this returns.
  virtual void notifyOperationDetached(Operation *op) {}

  /// `op` was spliced from `from` into `to` in one transfer (moveBefore,
  /// block merge, inlining, region hoist). `sameBlock` marks a reorder
  /// within one block — the moveOpUpInBlock case no listener sees today.
  virtual void notifyOperationMoved(Operation *op, Block *to, Block *from,
                                    bool sameBlock) {}

  /// The dialect-conversion driver is committing its replacement of `op`:
  /// `replacements` holds the final value standing in for each of `op`'s
  /// results (null where a result was dropped). Fired from
  /// ReplaceOperationRewrite::commit, at the one moment both sides of the
  /// pairing exist — conversion defers RAUW to finalization, so no other
  /// observation point ever sees this map. `op` is still linked when this
  /// fires; it is unlinked immediately after and erased during cleanup.
  virtual void notifyConversionReplaced(Operation *op,
                                        ArrayRef<Value> replacements) {}

  /// `operand` of its owner op was rewired from `oldValue` to `newValue`
  /// (already applied when this fires). Every operand mutation funnels
  /// through OpOperand::set — setOperand calls, rewriter modifications, and
  /// each use updated by a replaceAllUsesWith loop. A null `newValue` marks
  /// the drop that precedes bulk teardown (dropAllUses). Not covered:
  /// operand-list resizes (insert/eraseOperands), which change arity rather
  /// than rewire an existing use.
  virtual void notifyOperandChanged(OpOperand &operand, Value oldValue,
                                    Value newValue) {}

  /// `value`'s type was mutated in place, from `oldType` (already applied).
  /// Value::setType is the single funnel for in-place retyping.
  virtual void notifyValueTypeChanged(Value value, Type oldType) {}

  /// The discardable attribute dictionary of `op` was replaced:
  /// `oldAttrs` -> `newAttrs` (already applied). Fired from every writer of
  /// the dictionary — set/removeDiscardableAttr, setAttr/removeAttr's
  /// discardable branch, setDiscardableAttrs, and setAttrs.
  virtual void notifyOperationAttributesChanged(Operation *op,
                                                DictionaryAttr oldAttrs,
                                                DictionaryAttr newAttrs) {}

  /// An inherent attribute stored in `op`'s properties was set to
  /// `newValue` (null marks removal), from `oldValue` (null: was absent).
  /// Fired from setInherentAttr, the transient funnel of the properties
  /// migration. NOT a complete channel: code that mutates a property struct
  /// in place through getProperties() bypasses every hook — the one known
  /// unobservable mutation path.
  virtual void notifyOperationInherentAttrChanged(Operation *op,
                                                  StringAttr name,
                                                  Attribute oldValue,
                                                  Attribute newValue) {}

protected:
  IRMutationObserver() = default;
};

namespace detail {
/// Single process-global observer slot. Read on every op-list mutation with
/// memory_order_relaxed; written only from setActiveIRMutationObserver.
extern std::atomic<IRMutationObserver *> activeIRMutationObserver;
} // namespace detail

/// Install (or, with nullptr, remove) the process-global mutation observer.
/// Must be called while no IR mutations are in flight — in practice, before
/// the pass manager runs and after it finishes.
void setActiveIRMutationObserver(IRMutationObserver *observer);

/// The currently installed observer, or nullptr. Inline: this is the
/// hot-path guard.
inline IRMutationObserver *getActiveIRMutationObserver() {
  return detail::activeIRMutationObserver.load(std::memory_order_relaxed);
}

} // namespace mlir

#endif // MLIR_IR_IRMUTATIONOBSERVER_H

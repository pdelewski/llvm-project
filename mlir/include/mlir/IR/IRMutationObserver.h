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
#include "llvm/ADT/STLFunctionalExtras.h"

#include <atomic>

namespace mlir {
class Diagnostic;
class Block;
class Operation;
class Pattern;
class Region;

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

  /// A rewrite pattern's matchAndRewrite was entered / has returned — the
  /// TRANSACTION bracket: every mutation notified between a begin and its
  /// matching end belongs to that one pattern application. Applications
  /// nest (folding inside a pattern fires an inner bracket). With this, the
  /// observer interface is self-sufficient: events (ilist/operand/attr
  /// hooks), intent (replace/erase/commit/clone), and transactions — no
  /// actions-API handler required, and no exclusivity conflict with one.
  ///
  /// `root` is the op the pattern was offered. A bracket without it says that
  /// SOMETHING declined and not what it was asked about, which is the
  /// difference between a decline record and a statistic.
  virtual void notifyPatternBegin(const Pattern &pattern, Operation *root) {}
  virtual void notifyPatternEnd(const Pattern &pattern, bool succeeded) {}

  /// The enclosing pattern application stated why it refused —
  /// RewriterBase::notifyMatchFailure forwards here while an observer is
  /// installed. `reasonCallback` materializes the author's text on demand,
  /// so an observer that is not recording reasons pays nothing beyond the
  /// virtual call, and the many patterns that never state a reason cost
  /// nothing at all. This is the reason channel the bracket lacks: BEGIN
  /// says what op was offered, END says it refused, this says why.
  virtual void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) {}

  /// A rewriter is replacing `op`'s results with `replacements` and will
  /// erase it — RewriterBase::replaceOp, the DECLARED pairing of a greedy
  /// rewrite, fired at entry while both sides are live. This is the same
  /// claim notifyConversionReplaced carries for the conversion driver
  /// (whose ConversionPatternRewriter overrides replaceOp, so the two
  /// never double-fire): the pass itself says these values stand in for
  /// that op — intent, not reconstruction.
  virtual void notifyRewriterReplaced(Operation *op,
                                      ArrayRef<Value> replacements) {}

  /// A rewriter is erasing `op` with NO replacement — RewriterBase::eraseOp,
  /// the rewriter's declared "this became nothing". Completes the intent
  /// pair with notifyRewriterReplaced: every greedy-rewriter disposal is now
  /// testimony. Fires at entry, op fully alive; also fires on the eraseOp
  /// inside replaceOp (consumers see the op already claimed and no-op).
  /// ConversionPatternRewriter overrides eraseOp, so the deferred path
  /// stays with notifyConversionReplaced.
  virtual void notifyRewriterErased(Operation *op) {}

  /// `clone` was just created as a deep copy of `original`
  /// (Operation::clone — the funnel for ALL cloning, including each nested
  /// op cloned through Region::cloneInto, which fire individually). This is
  /// the IRMapping's correspondence recorded at the only moment it exists;
  /// clone provenance is otherwise thrown away and must be reconstructed.
  /// Fired after the clone is fully built (regions included), before the
  /// caller sees it.
  virtual void notifyOperationCloned(Operation *original, Operation *clone) {}

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

  /// A block argument was added to its owner block (already appended /
  /// inserted; `arg` knows its block and index).
  virtual void notifyBlockArgumentAdded(BlockArgument arg) {}

  /// `arg` is about to be erased from its owner block — still intact when
  /// this fires; destroyed immediately after.
  virtual void notifyBlockArgumentErased(BlockArgument arg) {}

  /// `block` was linked into a region's block list.
  virtual void notifyBlockAttached(Block *block) {}

  /// `block` was unlinked from its region's block list. Still alive here.
  virtual void notifyBlockDetached(Block *block) {}

  /// `block` was spliced from `from` into `to` in one transfer (region
  /// inlining, block reordering). `sameRegion` marks a reorder within one
  /// region.
  virtual void notifyBlockMoved(Block *block, Region *to, Region *from,
                                bool sameRegion) {}

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

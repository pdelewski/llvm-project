//===- LogicalResult.cpp - Recording where refusals come from -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Storage for the failure-origin vocabulary in LogicalResult.h.
//
// This file STORES and never calls out: no callback, no function pointer, no
// consumer code running inside `failure()`. A consumer enables recording,
// clears at a boundary it chose, and reads back.
//
// The log keeps the FIRST entries and counts the rest rather than wrapping.
// A ring would keep the newest, which for refusals is the wrong end: the
// earliest refusal in a scope is usually the cause and the later ones its
// consequences. Keeping the front also makes the order trivially
// chronological.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LogicalResult.h"

bool llvm::failureOriginHookEnabled = false;

namespace {
/// Per-thread bounded log. Thread-local because a multithreaded pass manager
/// may have several pipelines refusing things at once, and an origin belongs
/// to the thread that produced it.
struct OriginLog {
  static constexpr unsigned Capacity = 64;
  llvm::FailureOrigin Entries[Capacity];
  /// Total notes since the last clear -- NOT an index. `Notes > Capacity`
  /// is how truncation becomes visible instead of silent.
  unsigned Notes = 0;
};
} // namespace

static thread_local OriginLog TheLog;

void llvm::noteFailureOrigin(const char *File, unsigned Line,
                             const char *Reason, DeclineKind Kind) {
  OriginLog &Log = TheLog;
  if (Log.Notes < OriginLog::Capacity)
    Log.Entries[Log.Notes] = {File, Reason, Line, Kind};
  ++Log.Notes;
}

const llvm::FailureOrigin *llvm::getFailureOrigins(unsigned &Count) {
  const OriginLog &Log = TheLog;
  Count = Log.Notes < OriginLog::Capacity ? Log.Notes : OriginLog::Capacity;
  return Log.Entries;
}

unsigned llvm::getFailureOriginsDropped() {
  const OriginLog &Log = TheLog;
  return Log.Notes > OriginLog::Capacity ? Log.Notes - OriginLog::Capacity : 0;
}

void llvm::clearFailureOrigins() { TheLog.Notes = 0; }

//===- LogicalResult.h - Utilities for handling success/failure -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_LOGICALRESULT_H
#define LLVM_SUPPORT_LOGICALRESULT_H

#include <cassert>
#include <cstdint>
#include <optional>

namespace llvm {
/// This class represents an efficient way to signal success or failure. It
/// should be preferred over the use of `bool` when appropriate, as it avoids
/// all of the ambiguity that arises in interpreting a boolean result. This
/// class is marked as NODISCARD to ensure that the result is processed. Users
/// may explicitly discard a result by using `(void)`, e.g.
/// `(void)functionThatReturnsALogicalResult();`. Given the intended nature of
/// this class, it generally shouldn't be used as the result of functions that
/// very frequently have the result ignored. This class is intended to be used
/// in conjunction with the utility functions below.
struct [[nodiscard]] LogicalResult {
public:
  /// If isSuccess is true a `success` result is generated, otherwise a
  /// 'failure' result is generated.
  static LogicalResult success(bool IsSuccess = true) {
    return LogicalResult(IsSuccess);
  }

  /// If isFailure is true a `failure` result is generated, otherwise a
  /// 'success' result is generated.
  static LogicalResult failure(bool IsFailure = true) {
    return LogicalResult(!IsFailure);
  }

  /// Returns true if the provided LogicalResult corresponds to a success value.
  constexpr bool succeeded() const { return IsSuccess; }

  /// Returns true if the provided LogicalResult corresponds to a failure value.
  constexpr bool failed() const { return !IsSuccess; }

private:
  LogicalResult(bool IsSuccess) : IsSuccess(IsSuccess) {}

  /// Boolean indicating if this is a success result, if false this is a
  /// failure result.
  bool IsSuccess;
};

//===----------------------------------------------------------------------===//
// Failure origins
//===----------------------------------------------------------------------===//
//
// A compiler records what it does and, today, nothing about what it declines.
// The vocabulary below lets a refusal say where it was constructed and, where
// the author bothered to classify it, why -- at the cost of one branch inside
// `failure()`, and without touching any of its call sites: the default
// arguments are evaluated at the caller.
//
// Recording is opt-in at runtime (`failureOriginHookEnabled`). Nothing is
// recorded, allocated or formatted unless a consumer asks for it.

/// Branch hint, spelled locally on purpose: this header includes only C++
/// standard headers today, and llvm/Support/Compiler.h would drag in the
/// generated llvm/Config/llvm-config.h for every one of its many thousands of
/// transitive includers.
#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define LLVM_ORIGIN_UNLIKELY(x) __builtin_expect(static_cast<bool>(x), false)
#endif
#endif
#ifndef LLVM_ORIGIN_UNLIKELY
#define LLVM_ORIGIN_UNLIKELY(x) (x)
#endif

/// Default location arguments. `__builtin_FILE`/`__builtin_LINE` evaluate at
/// the call site, which is the whole point; where they are unavailable the
/// defaults degrade to "unknown" rather than to a wrong location.
#if defined(__has_builtin)
#if __has_builtin(__builtin_FILE) && __has_builtin(__builtin_LINE)
#define LLVM_ORIGIN_FILE __builtin_FILE()
#define LLVM_ORIGIN_LINE __builtin_LINE()
#endif
#endif
#ifndef LLVM_ORIGIN_FILE
#define LLVM_ORIGIN_FILE nullptr
#define LLVM_ORIGIN_LINE 0u
#endif

/// How an author classified a refusal. The classes are not cosmetic: each one
/// implies a different response from whoever reads the record.
enum class DeclineKind : uint8_t {
  /// A located `failure()` whose author assigned no class. Distinct from
  /// `Illegal`: it means "located, unclassified", never "legal".
  Unclassified,
  /// The transformation cannot apply -- illegal, or not representable. Nothing
  /// to do about it.
  Illegal,
  /// It could have applied; a heuristic chose otherwise. Tunable.
  Policy,
  /// It should apply and nobody has implemented it yet. Usually a TODO with an
  /// issue number sitting beside the branch.
  Unimplemented,
};

/// One recorded refusal. `File` and `Reason` point at string literals with
/// static storage duration -- entries never own their strings, so recording
/// costs a pointer store.
struct FailureOrigin {
  const char *File = nullptr;
  const char *Reason = nullptr;
  unsigned Line = 0;
  DeclineKind Kind = DeclineKind::Unclassified;
};

/// Off by default. A consumer sets this to record origins; while it is false
/// the cost of the vocabulary below is one load and one not-taken branch.
extern bool failureOriginHookEnabled;

/// Append to the calling thread's bounded log. Callers must check the flag.
void noteFailureOrigin(const char *File, unsigned Line, const char *Reason,
                       DeclineKind Kind);

/// The calling thread's recorded origins, oldest first. The log keeps the
/// FIRST entries and counts the rest, so order is always chronological and the
/// earliest refusal -- usually the interesting one -- is never the one lost.
const FailureOrigin *getFailureOrigins(unsigned &Count);

/// Origins pushed but not kept since the last clear. Truncation is reported,
/// never silent.
unsigned getFailureOriginsDropped();

/// Reset the calling thread's log. Consumers scope recording by clearing at a
/// known boundary; without a boundary an origin has no subject.
void clearFailureOrigins();

/// Per-call refusal capture, for a caller that must attribute a refusal to one
/// specific query rather than to the ambient log -- an `all_of` over operands,
/// say, where which operand refused is the answer. Recording into a sink does
/// NOT also touch the ambient log; the sink's owner decides what to do with it.
class DeclineSink {
public:
  static constexpr unsigned Capacity = 4;

  void note(const char *File, unsigned Line, const char *Reason,
            DeclineKind Kind) {
    if (Count < Capacity)
      Entries[Count] = {File, Reason, Line, Kind};
    ++Count;
  }

  const FailureOrigin *begin() const { return Entries; }
  const FailureOrigin *end() const {
    return Entries + (Count < Capacity ? Count : Capacity);
  }
  /// Total notes, which may exceed what was kept.
  unsigned notes() const { return Count; }
  unsigned dropped() const { return Count > Capacity ? Count - Capacity : 0; }
  bool empty() const { return Count == 0; }
  void clear() { Count = 0; }

private:
  FailureOrigin Entries[Capacity];
  unsigned Count = 0;
};

/// Refuse, with a reason, from a predicate that answers a question with `bool`.
/// Returns false so the call reads as `return declined(...)`. The bool return
/// is deliberate: encoding "the answer is no" as "the query failed" conflates
/// two different things.
inline bool declined(DeclineKind Kind, const char *Reason,
                     const char *File = LLVM_ORIGIN_FILE,
                     unsigned Line = LLVM_ORIGIN_LINE) {
  if (LLVM_ORIGIN_UNLIKELY(failureOriginHookEnabled))
    noteFailureOrigin(File, Line, Reason, Kind);
  return false;
}

/// As above, attributing to one caller's sink instead of the ambient log.
inline bool declined(DeclineSink *Why, DeclineKind Kind, const char *Reason,
                     const char *File = LLVM_ORIGIN_FILE,
                     unsigned Line = LLVM_ORIGIN_LINE) {
  if (Why)
    Why->note(File, Line, Reason, Kind);
  else if (LLVM_ORIGIN_UNLIKELY(failureOriginHookEnabled))
    noteFailureOrigin(File, Line, Reason, Kind);
  return false;
}

/// Utility function to generate a LogicalResult. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline LogicalResult success(bool IsSuccess = true) {
  return LogicalResult::success(IsSuccess);
}

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
///
/// The location defaults are evaluated at the CALL SITE, so every existing
/// `return failure();` in the tree becomes locatable without being edited.
/// A caller that is propagating rather than deciding -- or that lives in a
/// header, where a recorded location would name the header for every caller
/// alike -- should pass `nullptr, 0` to stay out of the log.
inline LogicalResult failure(bool IsFailure = true,
                             const char *File = LLVM_ORIGIN_FILE,
                             unsigned Line = LLVM_ORIGIN_LINE) {
  if (IsFailure && LLVM_ORIGIN_UNLIKELY(failureOriginHookEnabled))
    noteFailureOrigin(File, Line, /*Reason=*/nullptr,
                      DeclineKind::Unclassified);
  return LogicalResult::failure(IsFailure);
}

/// Refuse with a reason from a function that reports through LogicalResult and
/// has no rewriter in scope to call `notifyMatchFailure` on. Records the
/// caller's location, not this header's.
inline LogicalResult declinedFailure(DeclineKind Kind, const char *Reason,
                                     const char *File = LLVM_ORIGIN_FILE,
                                     unsigned Line = LLVM_ORIGIN_LINE) {
  if (LLVM_ORIGIN_UNLIKELY(failureOriginHookEnabled))
    noteFailureOrigin(File, Line, Reason, Kind);
  return LogicalResult::failure();
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a success value.
inline bool succeeded(LogicalResult Result) { return Result.succeeded(); }

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a failure value.
inline bool failed(LogicalResult Result) { return Result.failed(); }

/// This class provides support for representing a failure result, or a valid
/// value of type `T`. This allows for integrating with LogicalResult, while
/// also providing a value on the success path.
template <typename T> class [[nodiscard]] FailureOr : public std::optional<T> {
public:
  /// Allow constructing from a LogicalResult. The result *must* be a failure.
  /// Success results should use a proper instance of type `T`.
  FailureOr(LogicalResult Result) {
    assert(failed(Result) &&
           "success should be constructed with an instance of 'T'");
  }
  /// Bypasses `failure()`'s origin recording: a default-constructed
  /// FailureOr would otherwise name this header as every caller's site.
  FailureOr() : FailureOr(LogicalResult::failure()) {}
  FailureOr(T &&Y) : std::optional<T>(std::forward<T>(Y)) {}
  FailureOr(const T &Y) : std::optional<T>(Y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  FailureOr(const FailureOr<U> &Other)
      : std::optional<T>(failed(Other) ? std::optional<T>()
                                       : std::optional<T>(*Other)) {}

  operator LogicalResult() const { return success(has_value()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using std::optional<T>::operator bool;
  using std::optional<T>::has_value;
};

/// Wrap a value on the success path in a FailureOr of the same value type.
template <typename T,
          typename = std::enable_if_t<!std::is_convertible_v<T, bool>>>
inline auto success(T &&Y) {
  return FailureOr<std::decay_t<T>>(std::forward<T>(Y));
}

/// This class represents success/failure for parsing-like operations that find
/// it important to chain together failable operations with `||`.  This is an
/// extended version of `LogicalResult` that allows for explicit conversion to
/// bool.
///
/// This class should not be used for general error handling cases - we prefer
/// to keep the logic explicit with the `succeeded`/`failed` predicates.
/// However, traditional monadic-style parsing logic can sometimes get
/// swallowed up in boilerplate without this, so we provide this for narrow
/// cases where it is important.
///
class [[nodiscard]] ParseResult : public LogicalResult {
public:
  ParseResult(LogicalResult Result = success()) : LogicalResult(Result) {}

  /// Failure is true in a boolean context.
  constexpr explicit operator bool() const { return failed(); }
};
} // namespace llvm

#endif // LLVM_SUPPORT_LOGICALRESULT_H

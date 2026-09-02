//===- LogicalResultTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LogicalResult.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

/// A function that refuses through the plain, unannotated spelling. Its
/// recorded location must be THIS function's line, not LogicalResult.h's.
LogicalResult plainRefusal() { return failure(); }
constexpr unsigned PlainRefusalLine = __LINE__ - 1;

bool boolPredicate(bool Ok, DeclineSink *Why = nullptr) {
  if (!Ok)
    return declined(Why, DeclineKind::Illegal, "not ok");
  return true;
}
constexpr unsigned BoolPredicateLine = __LINE__ - 3;

struct Enabled {
  Enabled() {
    failureOriginHookEnabled = true;
    clearFailureOrigins();
  }
  ~Enabled() {
    failureOriginHookEnabled = false;
    clearFailureOrigins();
  }
};

TEST(LogicalResultOrigins, DisabledByDefaultRecordsNothing) {
  clearFailureOrigins();
  ASSERT_FALSE(failureOriginHookEnabled);
  EXPECT_TRUE(failed(plainRefusal()));
  unsigned Count = 1;
  getFailureOrigins(Count);
  EXPECT_EQ(Count, 0u);
}

TEST(LogicalResultOrigins, PlainFailureRecordsItsCallSite) {
  Enabled Guard;
  EXPECT_TRUE(failed(plainRefusal()));
  unsigned Count = 0;
  const FailureOrigin *Origins = getFailureOrigins(Count);
  ASSERT_EQ(Count, 1u);
  EXPECT_EQ(Origins[0].Line, PlainRefusalLine);
  EXPECT_NE(Origins[0].File, nullptr);
  // An unannotated failure() is located but unclassified -- which is a
  // different claim from "legal".
  EXPECT_EQ(Origins[0].Kind, DeclineKind::Unclassified);
  EXPECT_EQ(Origins[0].Reason, nullptr);
}

TEST(LogicalResultOrigins, SuccessRecordsNothing) {
  Enabled Guard;
  EXPECT_TRUE(succeeded(failure(/*IsFailure=*/false)));
  unsigned Count = 1;
  getFailureOrigins(Count);
  EXPECT_EQ(Count, 0u);
}

TEST(LogicalResultOrigins, DeclinedCarriesReasonAndClass) {
  Enabled Guard;
  EXPECT_FALSE(boolPredicate(false));
  unsigned Count = 0;
  const FailureOrigin *Origins = getFailureOrigins(Count);
  ASSERT_EQ(Count, 1u);
  EXPECT_EQ(Origins[0].Line, BoolPredicateLine);
  EXPECT_EQ(Origins[0].Kind, DeclineKind::Illegal);
  EXPECT_STREQ(Origins[0].Reason, "not ok");
}

TEST(LogicalResultOrigins, SinkTakesPrecedenceOverAmbientLog) {
  Enabled Guard;
  DeclineSink Why;
  EXPECT_FALSE(boolPredicate(false, &Why));
  EXPECT_EQ(Why.notes(), 1u);
  EXPECT_STREQ(Why.begin()->Reason, "not ok");
  // Recording into a sink must not also touch the ambient log, or the
  // caller's own attribution is double-counted.
  unsigned Count = 1;
  getFailureOrigins(Count);
  EXPECT_EQ(Count, 0u);
}

TEST(LogicalResultOrigins, TruncationIsCountedNotSilent) {
  Enabled Guard;
  const unsigned Pushes = 200;
  for (unsigned I = 0; I < Pushes; ++I)
    (void)plainRefusal();
  unsigned Count = 0;
  getFailureOrigins(Count);
  // The log keeps the FIRST entries -- the earliest refusal in a scope is
  // usually the cause -- and reports what it could not keep.
  EXPECT_GT(Count, 0u);
  EXPECT_LT(Count, Pushes);
  EXPECT_EQ(Count + getFailureOriginsDropped(), Pushes);
}

TEST(LogicalResultOrigins, DefaultConstructedFailureOrStaysOutOfTheLog) {
  Enabled Guard;
  FailureOr<int> Unset;
  EXPECT_TRUE(failed(Unset));
  // Otherwise every default-constructed FailureOr in the tree would report
  // LogicalResult.h as its origin.
  unsigned Count = 1;
  getFailureOrigins(Count);
  EXPECT_EQ(Count, 0u);
}

} // namespace

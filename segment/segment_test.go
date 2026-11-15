package segment

import (
	"errors"
	"math"
	"sync"
	"testing"
	"time"
)

func approxEqual(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %v, want NaN", got)
		}
		return
	}
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestEmptyResult(t *testing.T) {
	a, err := NewAnalyzer()
	if err != nil {
		t.Fatal(err)
	}
	r := a.Result()
	if len(r.Segments) != 0 {
		t.Fatalf("expected no segments, got %d", len(r.Segments))
	}
	if r.TotalCost != 0 {
		t.Fatalf("expected zero cost, got %v", r.TotalCost)
	}
	if len(a.EffectiveDates()) != 0 {
		t.Fatal("expected no effective dates")
	}
	if _, _, _, ok := r.BoundaryAt("2026-01-01"); ok {
		t.Fatal("unknown date must not resolve to a segment")
	}
	if len(r.Views()) != 0 {
		t.Fatal("expected no views")
	}
}

func TestInvalidConstructionAndInputs(t *testing.T) {
	if _, err := NewAnalyzer(WithPenalty(-1)); err == nil {
		t.Fatal("negative penalty must be rejected")
	}
	if _, err := NewAnalyzer(WithPenalty(math.NaN())); err == nil {
		t.Fatal("NaN penalty must be rejected")
	}
	if _, err := NewAnalyzer(WithPenalty(math.Inf(1))); err == nil {
		t.Fatal("infinite penalty must be rejected")
	}

	a, err := NewAnalyzer()
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Upsert("01/01/2026", "s", 1, 1); err == nil {
		t.Fatal("malformed date must be rejected")
	}
	if err := a.Upsert("2026-01-01", "", 1, 1); err == nil {
		t.Fatal("empty source must be rejected")
	}
	if err := a.Upsert("2026-01-01", "s", 1, math.NaN()); err == nil {
		t.Fatal("NaN value must be rejected")
	}
	// A first report may itself be a withdrawal: the date simply has no
	// effective value until a non-withdrawn version arrives.
	if err := a.Retract("2026-01-01", "s", 0); err != nil {
		t.Fatalf("initial withdrawal must be accepted, got %v", err)
	}
	if _, ok := a.EffectiveValue("2026-01-01"); ok {
		t.Fatal("withdrawn-only date must have no effective value")
	}
}

func TestSingleSegmentAndSSE(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(10))
	// Constant values always collapse into one segment of zero SSE.
	for i, d := range []string{"2026-01-01", "2026-01-02", "2026-01-03"} {
		if err := a.Upsert(d, "crawler", int64(i+1), 5); err != nil {
			t.Fatal(err)
		}
	}
	r := a.Result()
	if len(r.Segments) != 1 {
		t.Fatalf("expected one segment, got %d", len(r.Segments))
	}
	seg := r.Segments[0]
	if seg.Start != "2026-01-01" || seg.End != "2026-01-03" {
		t.Fatalf("unexpected boundaries %s..%s", seg.Start, seg.End)
	}
	approxEqual(t, seg.Mean, 5)
	approxEqual(t, seg.SSE, 0)
	approxEqual(t, seg.Cost, 10)
	approxEqual(t, r.TotalCost, 10)

	start, end, mean, ok := r.BoundaryAt("2026-01-02")
	if !ok || start != "2026-01-01" || end != "2026-01-03" || mean != 5 {
		t.Fatalf("unexpected BoundaryAt result: %v %v %v %v", start, end, mean, ok)
	}
	views := r.Views()
	if len(views) != 3 {
		t.Fatalf("expected 3 views, got %d", len(views))
	}
	for _, v := range views {
		if v.SegmentStart != "2026-01-01" || v.Mean != 5 {
			t.Fatalf("unexpected view %+v", v)
		}
	}
}

func TestSegmentationMatchesBruteForce(t *testing.T) {
	values := []float64{1, 2, 2, 10, 11, 10, 1, 2}
	dates := make([]string, len(values))
	a, _ := NewAnalyzer(WithPenalty(4))
	for i, v := range values {
		d := dateOf(i)
		dates[i] = d
		if err := a.Upsert(d, "s", int64(i+1), v); err != nil {
			t.Fatal(err)
		}
	}
	r := a.Result()

	// Independent DP over the same values to check the global optimum.
	wantCost := bruteForceDP(values, 4)
	approxEqual(t, r.TotalCost, wantCost)

	// Reconstructed segments must partition every date exactly once.
	if r.Segments[0].Start != dates[0] || r.Segments[len(r.Segments)-1].End != dates[len(dates)-1] {
		t.Fatal("segments do not cover the full date range")
	}
	seen := 0
	for _, seg := range r.Segments {
		li := indexOf(dates, seg.Start)
		ri := indexOf(dates, seg.End)
		if li > ri || li != seen {
			t.Fatalf("segment %+v leaves a gap", seg)
		}
		var s, q float64
		k := float64(ri - li + 1)
		for i := li; i <= ri; i++ {
			s += values[i]
			q += values[i] * values[i]
		}
		approxEqual(t, seg.Mean, s/k)
		approxEqual(t, seg.SSE, q-s*s/k)
		approxEqual(t, seg.Cost, seg.SSE+4)
		seen = ri + 1
	}
	if seen != len(dates) {
		t.Fatal("not all dates covered")
	}
}

func TestLateBackfillReusesPrefix(t *testing.T) {
	// First arrive day 1, 2 and 4; day 3 is reported late afterwards.
	a, _ := NewAnalyzer(WithPenalty(3))
	mustUpsert(t, a, "2026-01-01", "s", 1, 1)
	mustUpsert(t, a, "2026-01-02", "s", 1, 2)
	mustUpsert(t, a, "2026-01-04", "s", 1, 40)
	r1 := a.Result()
	if len(r1.Dates) != 3 {
		t.Fatalf("expected 3 effective dates, got %d", len(r1.Dates))
	}
	// The missing day is skipped, not zero-filled.
	if got := a.EffectiveDates(); len(got) != 3 || got[2] != "2026-01-04" {
		t.Fatalf("unexpected effective dates %v", got)
	}
	evalsBefore := a.EvalCount()

	// Day 3 arrives late: the prefix before it must keep its DP state.
	mustUpsert(t, a, "2026-01-03", "s", 1, 3)
	r2 := a.Result()
	wantDates := []string{"2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"}
	if len(r2.Dates) != 4 {
		t.Fatalf("expected 4 dates, got %v", r2.Dates)
	}
	for i, d := range wantDates {
		if r2.Dates[i] != d {
			t.Fatalf("date %d: got %s, want %s", i, r2.Dates[i], d)
		}
	}
	// The late date changed the middle of the sequence; only rows from its
	// position onward were evaluated (rows 3 and 4: 3+4 evaluations).
	if got := a.EvalCount() - evalsBefore; got != 7 {
		t.Fatalf("expected 7 suffix evaluations, got %d", got)
	}
	// The result must equal a fresh solve over the completed sequence.
	fresh, _ := NewAnalyzer(WithPenalty(3))
	for i, v := range []float64{1, 2, 3, 40} {
		mustUpsert(t, fresh, dateOf(i), "s", int64(i+1), v)
	}
	approxEqual(t, r2.TotalCost, fresh.Result().TotalCost)
}

func TestNewEndOfDayReportIsLinearExtension(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(2))
	for i := 0; i < 5; i++ {
		mustUpsert(t, a, dateOf(i), "s", int64(i+1), float64(i))
	}
	a.Result()
	evalsBefore := a.EvalCount()

	mustUpsert(t, a, dateOf(5), "s", 1, 50)
	a.Result()
	// Appending a new last date evaluates exactly the single new DP row,
	// whose row index equals the number of effective values (6).
	if got := a.EvalCount() - evalsBefore; got != 6 {
		t.Fatalf("expected 6 evaluations for one appended row, got %d", got)
	}
}

func TestRevisionRecomputesOnlySuffix(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(2))
	for i, v := range []float64{1, 1, 1, 100, 100} {
		mustUpsert(t, a, dateOf(i), "s", 1, v)
	}
	first := a.Result()

	// Revise day 2 (index 1): everything after it is recomputed, the dp row
	// for day 1 stays valid.
	mustUpsert(t, a, dateOf(1), "s", 2, 80)
	evalsBefore := a.EvalCount()
	second := a.Result()
	// Rows 2..5 evaluated: 2+3+4+5 = 14.
	if got := a.EvalCount() - evalsBefore; got != 14 {
		t.Fatalf("expected 14 suffix evaluations, got %d", got)
	}

	fresh, _ := NewAnalyzer(WithPenalty(2))
	want := []float64{1, 80, 1, 100, 100}
	for i, v := range want {
		mustUpsert(t, fresh, dateOf(i), "s", 1, v)
	}
	approxEqual(t, second.TotalCost, fresh.Result().TotalCost)
	if first.TotalCost == second.TotalCost {
		t.Fatal("revision changing an effective value must change the optimum")
	}
}

func TestNoopRevisionDoesNotRecompute(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(2))
	mustUpsert(t, a, dateOf(0), "s", 1, 7)
	a.Result()
	evals := a.EvalCount()

	// Same version, same value: a duplicate report changes nothing.
	if err := a.Upsert(dateOf(0), "s", 1, 7); err != nil {
		t.Fatalf("identical duplicate must be accepted, got %v", err)
	}
	// Same version, different value: rejected.
	if err := a.Upsert(dateOf(0), "s", 1, 9); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("expected ErrVersionConflict, got %v", err)
	}
	// Older version: rejected.
	if err := a.Upsert(dateOf(0), "s", 0, 9); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("expected ErrVersionConflict, got %v", err)
	}
	// Stale withdrawal: rejected.
	if err := a.Retract(dateOf(0), "s", 1); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("expected ErrVersionConflict, got %v", err)
	}
	if a.EvalCount() != evals {
		t.Fatal("rejected/duplicate reports must not trigger any recomputation")
	}
	if v, ok := a.EffectiveValue(dateOf(0)); !ok || v != 7 {
		t.Fatalf("effective value changed unexpectedly: %v %v", v, ok)
	}
	if a.Result() == nil {
		t.Fatal("cached result must remain available")
	}
}

func TestWithdrawalAndSourceFallback(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(1), WithSourceOrder("official", "crawler"))
	mustUpsert(t, a, dateOf(0), "crawler", 1, 10)
	mustUpsert(t, a, dateOf(0), "official", 1, 12)
	if v, ok := a.EffectiveValue(dateOf(0)); !ok || v != 12 {
		t.Fatalf("higher-priority source must win, got %v", v)
	}

	// Official source revises.
	mustUpsert(t, a, dateOf(0), "official", 2, 14)
	if v, _ := a.EffectiveValue(dateOf(0)); v != 14 {
		t.Fatalf("revision not picked up, got %v", v)
	}

	// Official withdraws: the crawler value becomes effective again.
	if err := a.Retract(dateOf(0), "official", 3); err != nil {
		t.Fatal(err)
	}
	if v, ok := a.EffectiveValue(dateOf(0)); !ok || v != 10 {
		t.Fatalf("expected fallback value 10, got %v (ok=%v)", v, ok)
	}

	// Crawler withdraws too: the date disappears entirely, never becoming zero.
	if err := a.Retract(dateOf(0), "crawler", 2); err != nil {
		t.Fatal(err)
	}
	if _, ok := a.EffectiveValue(dateOf(0)); ok {
		t.Fatal("fully withdrawn date must have no effective value")
	}
	if dates := a.EffectiveDates(); len(dates) != 0 {
		t.Fatalf("expected no effective dates, got %v", dates)
	}
	if r := a.Result(); len(r.Segments) != 0 || r.TotalCost != 0 {
		t.Fatalf("expected empty result, got %+v", r)
	}

	// A crawler re-report at a new version brings the date back.
	mustUpsert(t, a, dateOf(0), "crawler", 3, 11)
	if v, ok := a.EffectiveValue(dateOf(0)); !ok || v != 11 {
		t.Fatalf("expected re-reported value 11, got %v", v)
	}
}

func TestDefaultSourceOrderIsLexicographic(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(0))
	mustUpsert(t, a, dateOf(0), "zeta", 1, 1)
	mustUpsert(t, a, dateOf(0), "alpha", 1, 2)
	if v, _ := a.EffectiveValue(dateOf(0)); v != 2 {
		t.Fatalf("without WithSourceOrder the lexicographically smallest source wins, got %v", v)
	}
}

func TestLowPriorityReportDoesNotChangeStatistics(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(1), WithSourceOrder("official"))
	mustUpsert(t, a, dateOf(0), "official", 1, 5)
	a.Result()
	evals := a.EvalCount()

	// A second source reporting the same date is accepted but, being lower
	// priority, cannot change the effective value or the segmentation.
	mustUpsert(t, a, dateOf(0), "backup", 1, 99)
	if v, _ := a.EffectiveValue(dateOf(0)); v != 5 {
		t.Fatalf("lower-priority source must not override, got %v", v)
	}
	if a.EvalCount() != evals {
		t.Fatal("an ineffective report must not trigger recomputation")
	}
}

func TestIneffectiveWithdrawalDoesNotRecompute(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(1), WithSourceOrder("official", "crawler"))
	mustUpsert(t, a, dateOf(0), "official", 1, 5)
	mustUpsert(t, a, dateOf(0), "crawler", 1, 99)
	snapshot := a.Result()
	evals := a.EvalCount()

	// Withdrawing the lower-priority source leaves the official value in
	// charge: the statistics and the cached snapshot must be untouched.
	if err := a.Retract(dateOf(0), "crawler", 2); err != nil {
		t.Fatal(err)
	}
	if v, ok := a.EffectiveValue(dateOf(0)); !ok || v != 5 {
		t.Fatalf("effective value changed unexpectedly: %v %v", v, ok)
	}
	if a.EvalCount() != evals {
		t.Fatal("ineffective withdrawal must not trigger recomputation")
	}
	if a.Result() != snapshot {
		t.Fatal("cached result snapshot must be retained")
	}
}

func TestReadReturnsCachedSnapshot(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(1))
	mustUpsert(t, a, dateOf(0), "s", 1, 1)
	mustUpsert(t, a, dateOf(1), "s", 1, 2)
	r1 := a.Result()
	evals := a.EvalCount()
	r2 := a.Result()
	r3 := a.Result()
	if r1 != r2 || r2 != r3 {
		t.Fatal("repeated reads must return the same cached snapshot")
	}
	if a.EvalCount() != evals {
		t.Fatal("reads must not run the dynamic program again")
	}
	// Callers must observe an immutable snapshot.
	if cap(r1.Segments) != len(r1.Segments) {
		t.Fatal("segment slice should have no spare capacity for safe sharing")
	}
}

func TestConcurrentAccess(t *testing.T) {
	a, _ := NewAnalyzer(WithPenalty(2), WithSourceOrder("s1", "s2"))
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		source := "s1"
		if g%2 == 0 {
			source = "s2"
		}
		wg.Add(1)
		go func(id int, src string) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				d := dateOf(i)
				// Version numbers need not match arrival order, so conflicts
				// are a legitimate outcome of the interleaving.
				if err := a.Upsert(d, src, int64(id*100+i+1), float64(i+id)); err != nil &&
					!errors.Is(err, ErrVersionConflict) {
					t.Error(err)
					return
				}
				_ = a.Result()
				_, _ = a.EffectiveValue(d)
				_ = a.EffectiveDates()
			}
		}(g, source)
	}
	wg.Wait()
	// Final result must be internally consistent.
	r := a.Result()
	if len(r.Dates) != 50 {
		t.Fatalf("expected 50 effective dates, got %d", len(r.Dates))
	}
	var cost float64
	for _, seg := range r.Segments {
		cost += seg.Cost
	}
	approxEqual(t, cost, r.TotalCost)
}

func TestNegativeValuesAreSupported(t *testing.T) {
	// Missing values must not be zero-filled, but genuine negative and zero
	// values are legitimate metric values.
	a, _ := NewAnalyzer(WithPenalty(0.5))
	values := []float64{-3, -3, 0, 5, 5}
	for i, v := range values {
		mustUpsert(t, a, dateOf(i), "s", 1, v)
	}
	r := a.Result()
	approxEqual(t, r.TotalCost, bruteForceDP(values, 0.5))
	for _, seg := range r.Segments {
		if seg.SSE < -1e-12 {
			t.Fatalf("negative SSE %v", seg.SSE)
		}
	}
}

func mustUpsert(t *testing.T, a *Analyzer, date, source string, version int64, value float64) {
	t.Helper()
	if err := a.Upsert(date, source, version, value); err != nil {
		t.Fatalf("Upsert(%s, %s, %d, %v): %v", date, source, version, value, err)
	}
}

func dateOf(i int) string {
	// Fixed month; day values past January roll over into February safely.
	return time.Date(2026, time.January, 1+i, 0, 0, 0, 0, time.UTC).Format(dateLayout)
}

func indexOf(dates []string, d string) int {
	for i, v := range dates {
		if v == d {
			return i
		}
	}
	return -1
}

// bruteForceDP is an independent textbook implementation used as an oracle.
func bruteForceDP(values []float64, penalty float64) float64 {
	n := len(values)
	sum := make([]float64, n+1)
	sumSq := make([]float64, n+1)
	for i, v := range values {
		sum[i+1] = sum[i] + v
		sumSq[i+1] = sumSq[i] + v*v
	}
	dp := make([]float64, n+1)
	for i := 1; i <= n; i++ {
		dp[i] = math.Inf(1)
		for j := 0; j < i; j++ {
			k := float64(i - j)
			s := sum[i] - sum[j]
			sse := sumSq[i] - sumSq[j] - s*s/k
			if sse < 0 {
				sse = 0
			}
			if c := dp[j] + penalty + sse; c < dp[i] {
				dp[i] = c
			}
		}
	}
	return dp[n]
}

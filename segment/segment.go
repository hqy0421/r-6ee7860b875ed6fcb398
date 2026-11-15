// Package segment maintains an online piecewise-constant segmentation of a
// daily business metric whose values may be reported late, revised or
// withdrawn by several sources.
//
// Every reported value is a versioned observation for a (business date,
// source) pair. For each business date the value of the highest-priority
// source whose latest version is not withdrawn is the effective value.
// Dates without any effective value are simply absent from the analysis:
// a missing report is never turned into a zero.
//
// The effective values, ordered by business date, are partitioned into
// constant-mean segments. A partition of n values costs
//
//	sum over segments of (penalty + sum of squared errors around the mean)
//
// and the Analyzer keeps the partition with the global minimum cost.
// Optimal substructure allows a dynamic program:
//
//	dp[i] = min over j < i of dp[j] + penalty + sse(values[j:i])
//
// Observations arrive over time. A new report for the newest business date
// only extends the sequence, so every previously computed dp entry is
// reused. A late backfill, a revision or a withdrawal that first changes an
// earlier business date invalidates only the suffix starting at that date:
// dp entries before it describe an unchanged prefix and stay valid. Reading
// the result never triggers a full re-solve; at most the affected suffix is
// evaluated, and repeated reads return the cached snapshot.
package segment

import (
	"errors"
	"math"
	"sort"
	"sync"
	"time"
)

// dateLayout is the accepted business date format (RFC3339 date). Lexical
// ordering of strings in this layout is chronological ordering.
const dateLayout = "2006-01-02"

// ErrVersionConflict is returned when an observation carries a version that
// cannot supersede the version already stored for the same (date, source).
// A stored version is immutable: an observation with the same version but a
// different value or withdrawn flag, or any smaller version, is rejected and
// leaves the statistics untouched.
var ErrVersionConflict = errors.New("segment: observation version cannot supersede the stored version")

// Observation is a versioned value reported by one source for one business
// date. Version must increase with every new report of the same (Date,
// Source) pair; the largest stored version wins.
type Observation struct {
	// Date is the business date the value refers to, formatted "2006-01-02".
	Date string
	// Source identifies the reporting system.
	Source string
	// Version is the monotonically increasing report version per (Date, Source).
	Version int64
	// Value is the reported metric value. It is ignored when Withdrawn is true.
	Value float64
	// Withdrawn marks the report as retracted: this source no longer provides
	// a value for Date at this version.
	Withdrawn bool
}

// Segment is one constant-mean interval of an optimal segmentation.
type Segment struct {
	// Start is the first business date belonging to the segment.
	Start string
	// End is the last business date belonging to the segment.
	End string
	// Mean is the arithmetic mean of the effective values in the segment.
	Mean float64
	// SSE is the sum of the squared errors of the segment's values around Mean.
	SSE float64
	// Cost is SSE plus the fixed per-segment penalty.
	Cost float64
}

// DateView is the segmentation information attributed to one business date.
type DateView struct {
	// Date is the business date.
	Date string
	// SegmentStart is the first date of the segment containing Date.
	SegmentStart string
	// SegmentEnd is the last date of the segment containing Date.
	SegmentEnd string
	// Mean is the mean of the segment containing Date.
	Mean float64
}

// Result is an immutable optimal segmentation snapshot over all currently
// effective dates.
type Result struct {
	// Segments are the optimal segments in chronological order.
	Segments []Segment
	// Dates are the effective business dates in chronological order.
	Dates []string
	// TotalCost is the global minimum cost, i.e. the sum of every Segment.Cost.
	TotalCost float64

	dateIndex map[string]int
	segOfDate []int // segOfDate[i] is the Segments index of Dates[i]
}

// BoundaryAt returns the start and end date of the segment containing date
// and its mean. ok is false when date has no effective value.
func (r *Result) BoundaryAt(date string) (start, end string, mean float64, ok bool) {
	i, ok := r.dateIndex[date]
	if !ok {
		return "", "", 0, false
	}
	s := r.Segments[r.segOfDate[i]]
	return s.Start, s.End, s.Mean, true
}

// Views returns one DateView per effective date in chronological order.
func (r *Result) Views() []DateView {
	views := make([]DateView, len(r.Dates))
	for i, d := range r.Dates {
		s := r.Segments[r.segOfDate[i]]
		views[i] = DateView{
			Date:         d,
			SegmentStart: s.Start,
			SegmentEnd:   s.End,
			Mean:         s.Mean,
		}
	}
	return views
}

// record is the latest stored version of one (date, source) pair.
type record struct {
	version   int64
	value     float64
	withdrawn bool
}

// Analyzer ingests versioned observations and serves the optimal
// segmentation of the currently effective daily values. It is safe for
// concurrent use.
type Analyzer struct {
	mu      sync.RWMutex
	penalty float64
	// rank maps an explicitly ordered source to its priority; smaller wins.
	// Sources absent from rank share the lowest priority tier and break ties
	// lexicographically.
	rank map[string]int

	// records holds the newest record per source for every known business date.
	records map[string]map[string]*record
	// dates is the sorted set of every business date ever observed, including
	// dates that currently have no effective value.
	dates []string

	// dp[i] is the minimum cost of partitioning the first i effective values
	// of the last solved sequence; cut[i] is the start index of the last
	// segment in that optimal partition.
	dp  []float64
	cut []int

	// firstChange is the earliest business date whose effective value changed
	// since the last solve; "" means the cached result is up to date.
	firstChange string
	result      *Result

	// evals counts segment-cost evaluations performed by the dynamic program.
	// It exists for tests and diagnostics; reads never increment it.
	evals int64
}

// Option configures an Analyzer at construction.
type Option func(*Analyzer)

// WithPenalty sets the fixed cost added for every segment of a partition.
// It must be non-negative and finite. A zero penalty yields one segment per
// value; larger penalties merge adjacent values into fewer segments.
func WithPenalty(penalty float64) Option {
	return func(a *Analyzer) {
		a.penalty = penalty
	}
}

// WithSourceOrder defines the source priority used when several sources
// report the same business date. Sources listed earlier win; sources not
// listed share the lowest priority tier and break ties lexicographically.
func WithSourceOrder(sources ...string) Option {
	return func(a *Analyzer) {
		a.rank = make(map[string]int, len(sources))
		for i, s := range sources {
			a.rank[s] = i
		}
	}
}

// NewAnalyzer creates an Analyzer with the given options.
func NewAnalyzer(options ...Option) (*Analyzer, error) {
	a := &Analyzer{
		records: make(map[string]map[string]*record),
		rank:    make(map[string]int),
	}
	for _, opt := range options {
		opt(a)
	}
	if math.IsNaN(a.penalty) || math.IsInf(a.penalty, 0) || a.penalty < 0 {
		return nil, errors.New("segment: penalty must be a finite, non-negative number")
	}
	return a, nil
}

// Upsert stores value for business date as reported by source at version.
// It supports first reports, late backfills and revisions. Observations that
// do not supersede the stored version are rejected with ErrVersionConflict
// and never change the statistics.
func (a *Analyzer) Upsert(date, source string, version int64, value float64) error {
	return a.apply(Observation{
		Date:    date,
		Source:  source,
		Version: version,
		Value:   value,
	})
}

// Retract withdraws the value that source reported for business date, using
// version as the new report version. A withdrawal only takes effect when
// version is strictly greater than the stored version.
func (a *Analyzer) Retract(date, source string, version int64) error {
	return a.apply(Observation{
		Date:      date,
		Source:    source,
		Version:   version,
		Withdrawn: true,
	})
}

// apply stores a generic observation. See Upsert and Retract for semantics.
// Callers must not hold any lock; apply takes the write lock itself.
func (a *Analyzer) apply(o Observation) error {
	if _, err := time.Parse(dateLayout, o.Date); err != nil {
		return errors.New("segment: invalid business date, expected format " + dateLayout)
	}
	if o.Source == "" {
		return errors.New("segment: source must not be empty")
	}
	if math.IsNaN(o.Value) || math.IsInf(o.Value, 0) {
		return errors.New("segment: value must be finite")
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	bySource, ok := a.records[o.Date]
	if !ok {
		bySource = make(map[string]*record)
		a.records[o.Date] = bySource
		a.insertDate(o.Date)
	}
	stored, exists := bySource[o.Source]
	if exists {
		if o.Version < stored.version {
			return ErrVersionConflict
		}
		if o.Version == stored.version &&
			(stored.withdrawn != o.Withdrawn || (!o.Withdrawn && stored.value != o.Value)) {
			return ErrVersionConflict
		}
		if o.Version == stored.version {
			// Duplicate of the stored report: nothing changes.
			return nil
		}
	}

	// Only invalidate the cached segmentation when the effective value of
	// the date actually changes: a lower-priority source or a withdrawal that
	// leaves another source in charge must not perturb the statistics.
	oldValue, oldOK := a.effectiveValueLocked(o.Date)
	bySource[o.Source] = &record{
		version:   o.Version,
		value:     o.Value,
		withdrawn: o.Withdrawn,
	}
	newValue, newOK := a.effectiveValueLocked(o.Date)
	if oldOK != newOK || (oldOK && oldValue != newValue) {
		a.invalidate(o.Date)
	}
	return nil
}

// insertDate keeps the sorted set of known dates. Callers must hold mu.
func (a *Analyzer) insertDate(date string) {
	i := sort.SearchStrings(a.dates, date)
	if i < len(a.dates) && a.dates[i] == date {
		return
	}
	a.dates = append(a.dates, "")
	copy(a.dates[i+1:], a.dates[i:])
	a.dates[i] = date
}

// invalidate marks the cached result stale and remembers the earliest
// business date whose effective value may have changed. Callers must hold mu.
func (a *Analyzer) invalidate(date string) {
	if a.firstChange == "" || date < a.firstChange {
		a.firstChange = date
	}
	a.result = nil
}

// EffectiveValue returns the currently effective value of date: the value of
// the highest-priority source whose latest version is not withdrawn. Missing
// dates and dates whose sources all withdrew report ok=false rather than zero.
func (a *Analyzer) EffectiveValue(date string) (value float64, ok bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.effectiveValueLocked(date)
}

// effectiveValueLocked computes the effective value of one date. Callers must
// hold at least a read lock.
func (a *Analyzer) effectiveValueLocked(date string) (float64, bool) {
	bySource, ok := a.records[date]
	if !ok {
		return 0, false
	}
	var bestSource string
	var bestRecord *record
	for source, rec := range bySource {
		if rec.withdrawn {
			continue
		}
		if bestRecord == nil || a.sourceLess(source, bestSource) {
			bestSource = source
			bestRecord = rec
		}
	}
	if bestRecord == nil {
		return 0, false
	}
	return bestRecord.value, true
}

// sourceLess reports whether source s has higher priority than source t.
func (a *Analyzer) sourceLess(s, t string) bool {
	rs, okS := a.rank[s]
	if !okS {
		rs = len(a.rank)
	}
	rt, okT := a.rank[t]
	if !okT {
		rt = len(a.rank)
	}
	if rs != rt {
		return rs < rt
	}
	return s < t
}

// EffectiveDates returns the business dates that currently have an effective
// value, in chronological order. The returned slice is a copy.
func (a *Analyzer) EffectiveDates() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	dates := make([]string, 0, len(a.dates))
	for _, d := range a.dates {
		if _, ok := a.effectiveValueLocked(d); ok {
			dates = append(dates, d)
		}
	}
	return dates
}

// Result returns the optimal segmentation over the currently effective
// values. The returned pointer is an immutable snapshot: it is cached and
// returned directly on repeated reads, and building it never re-solves the
// unchanged prefix.
func (a *Analyzer) Result() *Result {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.result == nil {
		a.result = a.solve()
	}
	return a.result
}

// EvalCount returns the number of segment-cost evaluations the dynamic
// program has performed. It is intended for tests verifying incremental
// recomputation; result reads do not add evaluations once the result is cached.
func (a *Analyzer) EvalCount() int64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.evals
}

// effectivePoint is one business date with its effective value.
type effectivePoint struct {
	date  string
	value float64
}

// solve runs the dynamic program over the suffix affected by the changes
// since the previous solve and reconstructs the optimal Result. Callers must
// hold mu for writing. The dp prefix before the first changed date is reused
// as-is, so the cost of a new end-of-day report is linear in the history
// instead of quadratic.
func (a *Analyzer) solve() *Result {
	points := make([]effectivePoint, 0, len(a.dates))
	for _, d := range a.dates {
		if v, ok := a.effectiveValueLocked(d); ok {
			points = append(points, effectivePoint{date: d, value: v})
		}
	}
	n := len(points)

	// start is the first position of the effective sequence whose optimal
	// subproblem may have changed. Everything before it is unchanged.
	start := 0
	if a.firstChange != "" {
		start = sort.Search(n, func(i int) bool { return points[i].date >= a.firstChange })
	}

	// Prefix sums of values and squared values give segment SSE in O(1).
	sum := make([]float64, n+1)
	sumSq := make([]float64, n+1)
	for i, p := range points {
		sum[i+1] = sum[i] + p.value
		sumSq[i+1] = sumSq[i] + p.value*p.value
	}
	segmentSSE := func(l, r int) float64 {
		k := float64(r - l)
		s := sum[r] - sum[l]
		q := sumSq[r] - sumSq[l]
		sse := q - s*s/k
		// Rounding can produce tiny negative values for near-constant segments.
		if sse < 0 {
			sse = 0
		}
		return sse
	}

	// Resize the DP tables while preserving the still-valid prefix.
	if len(a.dp) < n+1 {
		a.dp = append(a.dp, make([]float64, n+1-len(a.dp))...)
		a.cut = append(a.cut, make([]int, n+1-len(a.cut))...)
	} else {
		a.dp = a.dp[:n+1]
		a.cut = a.cut[:n+1]
	}

	for i := start + 1; i <= n; i++ {
		best := math.Inf(1)
		bestCut := -1
		for j := 0; j < i; j++ {
			cost := a.dp[j] + a.penalty + segmentSSE(j, i)
			if cost < best {
				best = cost
				bestCut = j
			}
		}
		a.dp[i] = best
		a.cut[i] = bestCut
		a.evals += int64(i)
	}

	result := &Result{
		Segments:  a.reconstruct(points, sum, sumSq),
		Dates:     make([]string, n),
		TotalCost: 0,
		dateIndex: make(map[string]int, n),
		segOfDate: make([]int, n),
	}
	for i, p := range points {
		result.Dates[i] = p.date
		result.dateIndex[p.date] = i
	}
	if n > 0 {
		result.TotalCost = a.dp[n]
	}
	for si, seg := range result.Segments {
		left := sort.SearchStrings(result.Dates, seg.Start)
		right := sort.SearchStrings(result.Dates, seg.End)
		for i := left; i <= right; i++ {
			result.segOfDate[i] = si
		}
	}

	a.firstChange = ""
	return result
}

// reconstruct walks the cut table backwards to materialize the optimal
// segments, recomputing each segment's mean and SSE from the prefix sums.
func (a *Analyzer) reconstruct(points []effectivePoint, sum, sumSq []float64) []Segment {
	n := len(points)
	if n == 0 {
		return []Segment{}
	}
	segments := make([]Segment, 0)
	for k := n; k > 0; {
		j := a.cut[k]
		count := float64(k - j)
		s := sum[k] - sum[j]
		q := sumSq[k] - sumSq[j]
		mean := s / count
		sse := q - s*s/count
		if sse < 0 {
			sse = 0
		}
		segments = append(segments, Segment{
			Start: points[j].date,
			End:   points[k-1].date,
			Mean:  mean,
			SSE:   sse,
			Cost:  sse + a.penalty,
		})
		k = j
	}
	for left, right := 0, len(segments)-1; left < right; left, right = left+1, right-1 {
		segments[left], segments[right] = segments[right], segments[left]
	}
	return segments
}

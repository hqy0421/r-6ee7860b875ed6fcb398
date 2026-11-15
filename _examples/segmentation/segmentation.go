// Example: feed versioned daily metric reports scraped with Colly into a
// segment.Analyzer. The scraped pages arrive in an order that exercises late
// backfills, revisions, lower-priority duplicates and withdrawals; the
// analyzer keeps the optimal piecewise-constant segmentation incrementally.
package main

import (
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"

	"github.com/gocolly/colly/v2"
	"github.com/gocolly/colly/v2/segment"
)

// reportPage is one scraped report, intentionally ordered by arrival time
// rather than by business date.
type reportPage struct {
	path      string
	date      string
	source    string
	version   int64
	value     float64
	withdrawn bool
}

func main() {
	pages := []reportPage{
		// Initial end-of-day reports; 2026-01-03 has not been reported yet.
		{path: "r1", date: "2026-01-01", source: "official", version: 1, value: 10},
		{path: "r2", date: "2026-01-02", source: "official", version: 1, value: 11},
		{path: "r3", date: "2026-01-04", source: "official", version: 1, value: 42},
		// Late backfill of the missing day.
		{path: "r4", date: "2026-01-03", source: "official", version: 1, value: 10.5},
		// Revision of an already reported day at a higher version.
		{path: "r5", date: "2026-01-01", source: "official", version: 2, value: 38},
		// A lower-priority source: accepted but never changes the statistics.
		{path: "r6", date: "2026-01-02", source: "crawler", version: 1, value: 99},
		// A stale crawler report that is subsequently retracted.
		{path: "r7", date: "2026-01-04", source: "crawler", version: 1, value: 100},
		{path: "r8", date: "2026-01-04", source: "crawler", version: 2, withdrawn: true},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b strings.Builder
		b.WriteString("<html><body>")
		if r.URL.Path == "/" {
			for _, p := range pages {
				b.WriteString(`<a href="/` + p.path + `">report</a>`)
			}
		} else {
			for _, p := range pages {
				if "/"+p.path != r.URL.Path {
					continue
				}
				b.WriteString(`<div class="report"`)
				b.WriteString(` data-date="` + p.date + `"`)
				b.WriteString(` data-source="` + p.source + `"`)
				b.WriteString(` data-version="` + strconv.FormatInt(p.version, 10) + `"`)
				b.WriteString(` data-value="` + strconv.FormatFloat(p.value, 'f', -1, 64) + `"`)
				if p.withdrawn {
					b.WriteString(` data-withdrawn="true"`)
				}
				b.WriteString("></div>")
			}
		}
		b.WriteString("</body></html>")
		fmt.Fprint(w, b.String())
	}))
	defer server.Close()

	analyzer, err := segment.NewAnalyzer(
		segment.WithPenalty(20),
		segment.WithSourceOrder("official", "crawler"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// The test server only serves the fixture pages above, so no domain
	// allow-list is needed.
	c := colly.NewCollector()

	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		e.Request.Visit(e.Attr("href"))
	})

	c.OnHTML("div.report", func(e *colly.HTMLElement) {
		version, err := strconv.ParseInt(e.Attr("data-version"), 10, 64)
		if err != nil {
			log.Fatalf("bad version: %v", err)
		}
		date, source := e.Attr("data-date"), e.Attr("data-source")

		if e.Attr("data-withdrawn") == "true" {
			if err := analyzer.Retract(date, source, version); err != nil {
				log.Fatalf("retract %s %s: %v", date, source, err)
			}
			fmt.Printf("withdrawn: %s from %s (v%d)\n", date, source, version)
			return
		}

		value, err := strconv.ParseFloat(e.Attr("data-value"), 64)
		if err != nil {
			log.Fatalf("bad value: %v", err)
		}
		switch err := analyzer.Upsert(date, source, version, value); {
		case err == nil:
			fmt.Printf("ingested:  %s from %-8s v%d = %g\n", date, source, version, value)
		case err == segment.ErrVersionConflict:
			fmt.Printf("ignored:   %s from %-8s v%d (stale version)\n", date, source, version)
		default:
			log.Fatalf("upsert %s: %v", date, err)
		}
	})

	if err := c.Visit(server.URL); err != nil {
		log.Fatal(err)
	}

	result := analyzer.Result()
	fmt.Println("\nOptimal segmentation of the currently effective values:")
	for _, view := range result.Views() {
		fmt.Printf("  %s -> segment [%s .. %s], mean %g\n",
			view.Date, view.SegmentStart, view.SegmentEnd, view.Mean)
	}
	fmt.Printf("\nsegments: %d  total cost: %.4f  dp evaluations: %d\n",
		len(result.Segments), result.TotalCost, analyzer.EvalCount())
}

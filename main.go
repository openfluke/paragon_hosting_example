package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/filesystem"
	htmleng "github.com/gofiber/template/html/v2"
	"github.com/openfluke/paragon/v3"
)

// ─────────────────────────────────────────────────────────────
// Embedded assets
// ─────────────────────────────────────────────────────────────

//go:embed web/templates/*.html
var templatesFS embed.FS

//go:embed web/static/*
var staticFS embed.FS

// ─────────────────────────────────────────────────────────────
// Server state
// ─────────────────────────────────────────────────────────────

type Server struct {
	NN         *paragon.Network[float32]
	InputW     int
	InputH     int
	ClassCount int
	ModelPath  string
	ModelName  string

	sem   chan struct{} // bound concurrent submissions
	gpuMu sync.Mutex    // serialize GPU if backend isn’t re-entrant

	inflight int64
	started  time.Time
}

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	modelPath := flag.String("model", "./models/mnist_model.json", "path to saved Paragon JSON model")
	maxGPU := flag.Int("maxgpu", 4, "max concurrent GPU submissions")
	flag.Parse()

	// 1) Load model (Paragon-style)
	nn, inW, inH, classes, err := loadParagonModel(*modelPath)
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}

	// 2) Mount on GPU once
	nn.WebGPUNative = true
	if err := nn.InitializeOptimizedGPU(); err != nil {
		log.Printf("WARN: WebGPU init failed: %v — falling back to CPU.", err)
		nn.WebGPUNative = false
	} else {
		log.Printf("GPU initialized.")
	}

	// 3) Warmup (zeros)
	if inW > 0 && inH > 0 {
		z := makeImage(inW, inH, 0)
		nn.Forward(z)
		_ = nn.ExtractOutput()
	}

	s := &Server{
		NN:         nn,
		InputW:     inW,
		InputH:     inH,
		ClassCount: classes,
		ModelPath:  filepath.Clean(*modelPath),
		ModelName:  filepath.Base(*modelPath),
		sem:        make(chan struct{}, *maxGPU),
		started:    time.Now(),
	}

	// 4) Views engine from embedded FS
	tmplFS, err := fs.Sub(templatesFS, "web/templates")
	if err != nil {
		log.Fatalf("embed FS sub mount: %v", err)
	}
	engine := htmleng.NewFileSystem(http.FS(tmplFS), ".html")
	engine.AddFunc("now", func() int { return time.Now().Year() })
	engine.Reload(true) // dev

	// 5) Fiber app
	app := fiber.New(fiber.Config{
		Views:        engine,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
	})

	// Static (embedded)
	app.Use("/static", filesystem.New(filesystem.Config{
		Root:       http.FS(staticFS),
		PathPrefix: "web/static",
		Browse:     false,
	}))

	// Pages
	app.Get("/", func(c *fiber.Ctx) error {
		return c.Render("home", fiber.Map{"Title": "Paragon Server · Home"}, "layout")
	})
	app.Get("/about", func(c *fiber.Ctx) error {
		return c.Render("about", fiber.Map{"Title": "About · Paragon Server"}, "layout")
	})
	app.Get("/test", func(c *fiber.Ctx) error {
		return c.Render("test", fiber.Map{"Title": "Load Test · Paragon Server"}, "layout")
	})

	// JSON service endpoints
	app.Get("/health", s.handleHealth)
	app.Get("/config", s.handleConfig)
	app.Post("/infer", s.handleInfer)              // one sample
	app.Post("/infer-batch", s.handleInferBatch)   // looped demo
	app.Post("/blast", s.handleBlast)              // N concurrent forwards
	app.Post("/save-session", s.handleSaveSession) // <-- NEW: persist session JSON

	// graceful shutdown
	go func() {
		sigc := make(chan os.Signal, 1)
		signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
		<-sigc
		log.Printf("Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if s.NN.WebGPUNative {
			s.NN.CleanupOptimizedGPU()
		}
		_ = app.ShutdownWithContext(ctx)
	}()

	log.Printf("Listening on %s", *addr)
	if err := app.Listen(*addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server error: %v", err)
	}
}

// ─────────────────────────────────────────────────────────────
// Paragon model loading that matches your project’s APIs
// ─────────────────────────────────────────────────────────────

func loadParagonModel(path string) (*paragon.Network[float32], int, int, int, error) {
	loaded, err := paragon.LoadNamedNetworkFromJSONFile(filepath.Clean(path))
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("LoadNamedNetworkFromJSONFile: %w", err)
	}
	tmp, ok := loaded.(*paragon.Network[float32])
	if !ok {
		return nil, 0, 0, 0, fmt.Errorf("model is not float32: %T", loaded)
	}

	// Derive shapes/activations from the loaded net’s layers
	shapes := make([]struct{ Width, Height int }, len(tmp.Layers))
	acts := make([]string, len(tmp.Layers))
	trains := make([]bool, len(tmp.Layers))
	for i, L := range tmp.Layers {
		shapes[i] = struct{ Width, Height int }{L.Width, L.Height}
		a := "linear"
		if L.Height > 0 && L.Width > 0 && L.Neurons[0][0] != nil {
			a = L.Neurons[0][0].Activation
		}
		acts[i], trains[i] = a, true
	}
	nn, err := paragon.NewNetwork[float32](shapes, acts, trains)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("NewNetwork: %w", err)
	}
	state, _ := tmp.MarshalJSONModel()
	if err := nn.UnmarshalJSONModel(state); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("UnmarshalJSONModel: %w", err)
	}
	inW, inH := shapes[0].Width, shapes[0].Height
	last := shapes[len(shapes)-1]
	classes := last.Width * last.Height
	return nn, inW, inH, classes, nil
}

// ─────────────────────────────────────────────────────────────
// JSON endpoints
// ─────────────────────────────────────────────────────────────

func (s *Server) handleHealth(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"status":   "ok",
		"uptime_s": time.Since(s.started).Seconds(),
		"inflight": atomic.LoadInt64(&s.inflight),
		"gpu":      s.NN.WebGPUNative,
	})
}

func (s *Server) handleConfig(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"input":     []int{s.InputW, s.InputH},
		"classes":   s.ClassCount,
		"gpu":       s.NN.WebGPUNative,
		"model":     s.ModelName,
		"modelPath": s.ModelPath,
		"startedAt": s.started.UTC().Format(time.RFC3339Nano),
	})
}

type inferReq struct {
	Input []float64   `json:"input"` // flattened w*h in [0..1]
	Image [][]float64 `json:"image"` // h×w
}
type inferResp struct {
	TopIndex  int       `json:"top_index"`
	TopScore  float64   `json:"top_score"`
	Probs     []float64 `json:"probs"`
	UsedGPU   bool      `json:"used_gpu"`
	LatencyMs float64   `json:"latency_ms"`
	QueuedMs  float64   `json:"queued_ms"`
	InFlight  int64     `json:"inflight"`
	When      time.Time `json:"when"`
}

func (s *Server) handleInfer(c *fiber.Ctx) error {
	var req inferReq
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, err.Error())
	}
	img, err := s.normalizeInput(req)
	if err != nil {
		return fiber.NewError(fiber.StatusBadRequest, err.Error())
	}

	startQ := time.Now()
	s.sem <- struct{}{}
	qDelay := time.Since(startQ)
	atomic.AddInt64(&s.inflight, 1)
	defer func() {
		<-s.sem
		atomic.AddInt64(&s.inflight, -1)
	}()

	start := time.Now()
	s.gpuMu.Lock()
	s.NN.Forward(img)
	out := s.NN.ExtractOutput() // []float64
	s.gpuMu.Unlock()

	idx := argmax64(out)
	return c.JSON(inferResp{
		TopIndex:  idx,
		TopScore:  out[idx],
		Probs:     out,
		UsedGPU:   s.NN.WebGPUNative,
		LatencyMs: durMs(time.Since(start)),
		QueuedMs:  durMs(qDelay),
		InFlight:  atomic.LoadInt64(&s.inflight),
		When:      time.Now(),
	})
}

type batchReq struct {
	Batch  [][]float64   `json:"batch"`  // N × (w*h)
	Images [][][]float64 `json:"images"` // N × h × w
}
type batchResp struct {
	TopIndices []int       `json:"top_indices"`
	TopScores  []float64   `json:"top_scores"`
	Probs      [][]float64 `json:"probs"`
	UsedGPU    bool        `json:"used_gpu"`
	LatencyMs  float64     `json:"latency_ms"`
	N          int         `json:"n"`
}

func (s *Server) handleInferBatch(c *fiber.Ctx) error {
	var req batchReq
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, err.Error())
	}
	var imgs [][][]float64
	switch {
	case len(req.Images) > 0:
		imgs = req.Images
	case len(req.Batch) > 0:
		for _, flat := range req.Batch {
			img, err := s.reshape(flat)
			if err != nil {
				return fiber.NewError(fiber.StatusBadRequest, err.Error())
			}
			imgs = append(imgs, img)
		}
	default:
		return fiber.NewError(fiber.StatusBadRequest, "provide 'images' or 'batch'")
	}

	s.sem <- struct{}{}
	defer func() { <-s.sem }()
	start := time.Now()

	s.gpuMu.Lock()
	topIdx := make([]int, len(imgs))
	topScores := make([]float64, len(imgs))
	probs := make([][]float64, len(imgs))
	for i := range imgs {
		s.NN.Forward(imgs[i])
		out := s.NN.ExtractOutput()
		idx := argmax64(out)
		topIdx[i], topScores[i], probs[i] = idx, out[idx], out
	}
	s.gpuMu.Unlock()

	return c.JSON(batchResp{
		TopIndices: topIdx,
		TopScores:  topScores,
		Probs:      probs,
		UsedGPU:    s.NN.WebGPUNative,
		LatencyMs:  durMs(time.Since(start)),
		N:          len(imgs),
	})
}

type blastReq struct {
	N     int       `json:"n"`
	Input []float64 `json:"input"`
}
type blastResp struct {
	Count    int         `json:"count"`
	Results  []inferResp `json:"results"`
	TotalMs  float64     `json:"total_ms"`
	Parallel int         `json:"parallel"`
}

func (s *Server) handleBlast(c *fiber.Ctx) error {
	var req blastReq
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, err.Error())
	}
	if req.N <= 0 || req.N > 2000 {
		return fiber.NewError(fiber.StatusBadRequest, "n must be 1..2000")
	}
	img, err := s.reshape(req.Input)
	if err != nil {
		return fiber.NewError(fiber.StatusBadRequest, err.Error())
	}

	start := time.Now()
	results := make([]inferResp, req.N)
	var wg sync.WaitGroup
	for i := 0; i < req.N; i++ {
		wg.Add(1)
		go func(ix int) {
			defer wg.Done()
			t0 := time.Now()
			s.sem <- struct{}{}
			qDelay := time.Since(t0)
			atomic.AddInt64(&s.inflight, 1)

			s.gpuMu.Lock()
			s.NN.Forward(img)
			out := s.NN.ExtractOutput()
			s.gpuMu.Unlock()

			idx := argmax64(out)
			results[ix] = inferResp{
				TopIndex:  idx,
				TopScore:  out[idx],
				Probs:     out,
				UsedGPU:   s.NN.WebGPUNative,
				LatencyMs: durMs(time.Since(t0)),
				QueuedMs:  durMs(qDelay),
				InFlight:  atomic.LoadInt64(&s.inflight),
				When:      time.Now(),
			}
			<-s.sem
			atomic.AddInt64(&s.inflight, -1)
		}(i)
	}
	wg.Wait()
	return c.JSON(blastResp{
		Count:    req.N,
		Results:  results,
		TotalMs:  durMs(time.Since(start)),
		Parallel: cap(s.sem),
	})
}

// NEW: save a full client session JSON to disk
func (s *Server) handleSaveSession(c *fiber.Ctx) error {
	var raw map[string]any
	if err := json.Unmarshal(c.Body(), &raw); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "invalid JSON")
	}
	if err := os.MkdirAll("./data/sessions", 0o755); err != nil {
		return fiber.NewError(fiber.StatusInternalServerError, err.Error())
	}
	ts := time.Now().UTC().Format("20060102T150405.000000000Z")
	fname := fmt.Sprintf("./data/sessions/%s_%s.json", ts, safeBase(s.ModelName))
	if err := os.WriteFile(fname, c.Body(), 0o644); err != nil {
		return fiber.NewError(fiber.StatusInternalServerError, err.Error())
	}
	return c.JSON(fiber.Map{
		"saved":   true,
		"path":    fname,
		"bytes":   len(c.Body()),
		"model":   s.ModelName,
		"created": ts,
	})
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

func makeImage(w, h int, val float64) [][]float64 {
	img := make([][]float64, h)
	for r := 0; r < h; r++ {
		row := make([]float64, w)
		for c := 0; c < w; c++ {
			row[c] = val
		}
		img[r] = row
	}
	return img
}

func (s *Server) reshape(flat []float64) ([][]float64, error) {
	if len(flat) != s.InputW*s.InputH {
		return nil, fmt.Errorf("flattened input must be length %d (got %d)", s.InputW*s.InputH, len(flat))
	}
	img := make([][]float64, s.InputH)
	for r := 0; r < s.InputH; r++ {
		row := make([]float64, s.InputW)
		for c := 0; c < s.InputW; c++ {
			v := flat[r*s.InputW+c]
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			row[c] = v
		}
		img[r] = row
	}
	return img, nil
}

func (s *Server) normalizeInput(req inferReq) ([][]float64, error) {
	switch {
	case len(req.Image) > 0:
		if len(req.Image) != s.InputH || len(req.Image[0]) != s.InputW {
			return nil, fmt.Errorf("image must be %dx%d (h×w)", s.InputH, s.InputW)
		}
		return req.Image, nil
	case len(req.Input) > 0:
		return s.reshape(req.Input)
	default:
		return nil, fmt.Errorf("provide 'image' or flattened 'input'")
	}
}

func argmax64(v []float64) int {
	if len(v) == 0 {
		return -1
	}
	bestI := 0
	best := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best, bestI = v[i], i
		}
	}
	return bestI
}

func durMs(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func safeBase(s string) string {
	b := filepath.Base(s)
	forbidden := []rune{'/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '}
	runes := []rune(b)
	for i, r := range runes {
		for _, f := range forbidden {
			if r == f {
				runes[i] = '_'
				break
			}
		}
	}
	return string(runes)
}

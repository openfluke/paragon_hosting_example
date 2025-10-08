# Paragon Server

A lightweight Go server for serving neural network inference using [Paragon](https://github.com/openfluke/paragon), a deterministic, cross-platform AI runtime. The server loads a trained model once (onto GPU via WebGPU or CPU fallback), then handles concurrent forward passes efficiently. Includes an embedded web UI for load testing, latency monitoring, and session capture.

![Screenshot of Load Test UI](https://via.placeholder.com/800x400?text=Paragon+Server+Load+Test) <!-- Replace with actual screenshot if available -->

## Features

- **Resident Model Loading**: Mounts weights and pipelines once; requests stream only input/output tensors.
- **GPU Acceleration**: Native WebGPU support for fast inference (falls back to CPU if unavailable).
- **Concurrent Handling**: Semaphore-limited parallelism (configurable via `-maxgpu` flag) to avoid overwhelming the backend.
- **JSON API**: Simple endpoints for single inference (`/infer`), batching (`/infer-batch`), and bursts (`/blast`).
- **Web UI**: Embedded single-page app for:
  - Home/About pages.
  - Load testing with client-side or server-side concurrency.
  - Real-time metrics (p50/p90/p99 latencies, inflight requests).
  - Session recording, JSON/CSV exports, and server-side saves.
- **Deterministic & Portable**: Mirrored activations ensure parity across CPU/GPU; no external deps beyond Go modules.
- **Graceful Shutdown**: Handles SIGINT/SIGTERM with cleanup.

## Quick Start

### Prerequisites

- Go 1.21+ (for `embed` support).
- A Paragon JSON model file (e.g., from training with Paragon's APIs). Example: MNIST classifier at `./models/mnist_model.json`.
- Optional: WebGPU-compatible browser/GPU for acceleration (Chrome/Edge with WebGPU flags enabled).

### Install & Run

1. Clone and build:

   ```
   git clone <repo>  # Or create a new dir with these files
   cd paragon-server
   go mod init paragon-server
   go mod tidy  # Pulls Fiber, Paragon, etc.
   go build -o server .
   ```

2. Run the server:

   ```
   ./server -model ./models/mnist_model.json -addr :8080 -maxgpu 4
   ```

   - `-model`: Path to your Paragon JSON model (required).
   - `-addr`: Listen address (default `:8080`).
   - `-maxgpu`: Max concurrent GPU submissions (default `4`).

3. Open in browser: [http://localhost:8080](http://localhost:8080)

The server will log GPU init status and warmup with zeros.

## Usage

### Web UI

- **Home (`/`)**: Overview of the server and features.
- **About (`/about`)**: Details on Paragon's design (resident model, streaming tensors).
- **Load Test (`/test`)**:
  - **Mode**:
    - _Client-side_: Simulates browser traffic with parallel `/infer` requests.
    - _Server-side_: Single `/blast` request for N concurrent forwards (uses Go goroutines).
  - **Requests (N)**: Number of inferences (1–2000).
  - **Parallel**: Browser concurrency for client mode (1–20 recommended).
  - **Input**: Zeros (deterministic) or random [0,1] pixels.
  - **Run**: Starts the test; watch the table for results.
  - **Sessions**: Start/stop to persist results across runs; export as JSON/CSV or save to `./data/sessions/`.
  - **Metrics**: Live percentiles, total time, and a bar chart of latencies.
  - **Recent Results**: Table with top class, scores, timings, and "View" buttons for full prob vectors.
  - **Health Poll**: Updates inflight requests every second.

Example session export (JSON):

```json
{
  "id": "sess_2025-10-08T12-00-00Z",
  "started_at": "2025-10-08T12:00:00.000Z",
  "model": "mnist_model.json",
  "gpu": true,
  "input": [28, 28],
  "results": [
    {
      "req_id": 0,
      "mode": "client",
      "started_at": "2025-10-08T12:00:01.000Z",
      "finished_at": "2025-10-08T12:00:01.050Z",
      "client_ms": 50.123,
      "server_ms": 45.2,
      "queue_ms": 0.1,
      "top_index": 7,
      "top_score": 0.9876,
      "probs": [0.01, 0.02, ..., 0.9876, ...]
    }
  ],
  "stopped_at": "2025-10-08T12:01:00.000Z"
}
```

### API Endpoints

All JSON-based. Assumes input shape from model (e.g., 28x28 for MNIST, flattened or 2D).

- **GET `/health`**: Server status.

  ```json
  { "status": "ok", "uptime_s": 123.45, "inflight": 2, "gpu": true }
  ```

- **GET `/config`**: Model info.

  ```json
  {
    "input": [28, 28],
    "classes": 10,
    "gpu": true,
    "model": "mnist_model.json",
    "modelPath": "/path/to/mnist_model.json",
    "startedAt": "2025-10-08T12:00:00Z"
  }
  ```

- **POST `/infer`**: Single inference.

  - Body: `{"input":[flattened pixels [0,1]]}` or `{"image":[[h x w array]]}`.
  - Response:
    ```json
    {"top_index":7,"top_score":0.9876,"probs":[...],"used_gpu":true,"latency_ms":45.2,"queued_ms":0.1,"inflight":1,"when":"2025-10-08T12:00:01Z"}
    ```

- **POST `/infer-batch`**: Batched inference (looped forwards).

  - Body: `{"batch":[[N x flattened]]}` or `{"images":[[N x h x w]]}`.
  - Response:
    ```json
    {"top_indices":[7,3,...],"top_scores":[0.9876,0.9123,...],"probs":[[...],...],"used_gpu":true,"latency_ms":120.5,"n":10}
    ```

- **POST `/blast`**: Concurrent burst (N goroutines).

  - Body: `{"n":100,"input":[flattened pixels]}`.
  - Response:
    ```json
    {"count":100,"results":[{inferResp},...],"total_ms":2500.0,"parallel":4}
    ```

- **POST `/save-session`**: Save UI session JSON to `./data/sessions/`.
  - Body: Full session object (as exported from UI).
  - Response: `{"saved":true,"path":"./data/sessions/2025-10-08T120000Z_mnist_model.json","bytes":2048,"model":"mnist_model.json","created":"2025-10-08T12:00:00Z"}`

Static assets served at `/static/*` (CSS/JS from embedded FS).

## Model Preparation

1. Train/export with Paragon (v3):

   ```go
   // Example: Save a simple MNIST net to JSON
   nn := paragon.NewNetwork[float32](shapes, acts, trains)
   // ... train ...
   state, _ := nn.MarshalJSONModel()
   jsonData := map[string]any{"model": "mnist", "state": state}
   jsonBytes, _ := json.Marshal(jsonData)
   os.WriteFile("models/mnist_model.json", jsonBytes, 0644)
   ```

2. Load via `loadParagonModel` in `main.go` – derives shapes/activations automatically.

For custom models, ensure output layer is flattened (classes = width \* height).

## Development

- **Templates**: Edit `web/templates/*.html`; `engine.Reload(true)` enables hot-reload.
- **Static**: Add to `web/static/` (embedded via `//go:embed`).
- **Testing**: Use `/test` UI or curl the API:
  ```
  curl -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{"input":[0.0,0.1,...]}' | jq '.top_index'
  ```
- **Profiling**: Add Fiber middleware or use `go tool pprof`.
- **Sessions Dir**: Saves to `./data/sessions/` (created on first save).

## Directory Structure

```
paragon-server/
├── main.go          # Server entrypoint
├── go.mod           # Modules (Fiber, Paragon, etc.)
├── models/          # Your JSON models
│   └── mnist_model.json
├── web/
│   ├── templates/
│   │   ├── layout.html
│   │   ├── home.html
│   │   ├── about.html
│   │   └── test.html
│   └── static/      # CSS/JS/images (embedded)
└── data/            # Runtime: sessions/ (created)
    └── sessions/
```

## Limitations

- WebGPU: Experimental; requires compatible hardware/browser. Fallback to CPU is automatic but slower.
- No auth/persistence beyond sessions.
- Model-specific: Input/output shapes from JSON; assumes float32 nets.
- Concurrency: GPU serialization via mutex (Paragon isn't re-entrant yet).

## License

APACHE2 License. See [LICENSE](LICENSE) for details. (c) 2025 OpenFluke. Built with ❤️ using Go, Fiber, and Paragon.

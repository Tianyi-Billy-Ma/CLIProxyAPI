package openai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	runtimeexecutor "github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
)

const testDiarizedJSON = `{"text":"hi there","segments":[{"start":0,"end":1.5,"text":"hi","speaker":"S01"},{"start":1.5,"end":2,"text":"there","speaker":"S02"}]}`

type capturedTranscription struct {
	path, auth, model, format, prompt, fileName string
	fileBytes                                   []byte
}

func newTranscriptionTestServer(t *testing.T, status int, body string) (*gin.Engine, func() []capturedTranscription) {
	t.Helper()
	var mu sync.Mutex
	var got []capturedTranscription
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture := capturedTranscription{path: r.URL.Path, auth: r.Header.Get("Authorization")}
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			t.Errorf("upstream parse multipart: %v", err)
		} else {
			capture.model = r.FormValue("model")
			capture.format = r.FormValue("response_format")
			capture.prompt = r.FormValue("prompt")
			if file, header, err := r.FormFile("file"); err == nil {
				capture.fileName = header.Filename
				capture.fileBytes, _ = io.ReadAll(file)
				_ = file.Close()
			}
		}
		mu.Lock()
		got = append(got, capture)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(upstream.Close)

	cfg := &internalconfig.Config{}
	providerKey := util.OpenAICompatibleProviderKey("spark")
	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(runtimeexecutor.NewOpenAICompatExecutor(providerKey, cfg))
	auth := &coreauth.Auth{
		ID:       "spark-asr-test-" + strings.ReplaceAll(t.Name(), "/", "_"),
		Provider: providerKey,
		Status:   coreauth.StatusActive,
		Attributes: map[string]string{
			"base_url":     upstream.URL + "/v1",
			"api_key":      "upstream-key",
			"compat_name":  "spark",
			"provider_key": providerKey,
		},
	}
	if _, err := manager.Register(context.Background(), auth); err != nil {
		t.Fatalf("manager.Register: %v", err)
	}
	reg := registry.GetGlobalRegistry()
	reg.RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: "SPARK-ASR", Object: "model", OwnedBy: "spark", Type: "openai-compatibility"}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() { reg.UnregisterClient(auth.ID) })

	h := NewOpenAIAPIHandler(handlers.NewBaseAPIHandlers(&cfg.SDKConfig, manager))
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.POST("/v1/audio/transcriptions", h.AudioTranscriptions)
	engine.GET("/v1/audio/transcriptions/:id", h.AudioTranscriptionRetrieve)
	return engine, func() []capturedTranscription {
		mu.Lock()
		defer mu.Unlock()
		return append([]capturedTranscription(nil), got...)
	}
}

func transcriptionForm(t *testing.T, fields map[string]string, withFile bool) (*bytes.Buffer, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			t.Fatal(err)
		}
	}
	if withFile {
		part, err := writer.CreateFormFile("file", "audio.m4a")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write([]byte("fake-m4a-bytes"))
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return &body, writer.FormDataContentType()
}

func doTranscription(engine *gin.Engine, method, path string, body io.Reader, contentType string, prefer string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, body)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if prefer != "" {
		req.Header.Set("Prefer", prefer)
	}
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)
	return rec
}

func TestAudioTranscriptionsSyncPassthrough(t *testing.T) {
	engine, captured := newTranscriptionTestServer(t, http.StatusOK, testDiarizedJSON)
	body, ct := transcriptionForm(t, map[string]string{"model": "SPARK-ASR", "response_format": "diarized_json", "prompt": "饭耳", "chunking_strategy": "auto"}, true)
	rec := doTranscription(engine, http.MethodPost, "/v1/audio/transcriptions", body, ct, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got := ojson.GetBytes(rec.Body.Bytes(), "segments.1.speaker").String(); got != "S02" {
		t.Fatalf("speaker = %q, body = %s", got, rec.Body.String())
	}
	calls := captured()
	if len(calls) != 1 {
		t.Fatalf("upstream calls = %d", len(calls))
	}
	c := calls[0]
	if c.path != "/v1/audio/transcriptions" || c.auth != "Bearer upstream-key" || c.model != "SPARK-ASR" ||
		c.format != "diarized_json" || c.prompt != "饭耳" || c.fileName != "audio.m4a" || string(c.fileBytes) != "fake-m4a-bytes" {
		t.Fatalf("unexpected upstream request: %+v", c)
	}
}

func TestAudioTranscriptionsValidation(t *testing.T) {
	engine, captured := newTranscriptionTestServer(t, http.StatusOK, testDiarizedJSON)
	noFile, ct := transcriptionForm(t, map[string]string{"model": "SPARK-ASR"}, false)
	noModel, ct2 := transcriptionForm(t, nil, true)
	cases := []struct {
		name        string
		body        io.Reader
		contentType string
	}{
		{"json body", strings.NewReader(`{"model":"SPARK-ASR"}`), "application/json"},
		{"missing file", noFile, ct},
		{"missing model", noModel, ct2},
	}
	for _, tc := range cases {
		rec := doTranscription(engine, http.MethodPost, "/v1/audio/transcriptions", tc.body, tc.contentType, "respond-async")
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: status = %d, body = %s", tc.name, rec.Code, rec.Body.String())
		}
	}
	if n := len(captured()); n != 0 {
		t.Fatalf("upstream calls = %d, want 0", n)
	}
}

func TestAudioTranscriptionsSyncUpstreamError(t *testing.T) {
	engine, _ := newTranscriptionTestServer(t, http.StatusBadRequest, `{"error":{"message":"response_format verbose_json not supported","type":"invalid_request_error"}}`)
	body, ct := transcriptionForm(t, map[string]string{"model": "SPARK-ASR", "response_format": "verbose_json"}, true)
	rec := doTranscription(engine, http.MethodPost, "/v1/audio/transcriptions", body, ct, "")
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "response_format") {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}

func waitTranscriptionJob(t *testing.T, engine *gin.Engine, id string) *httptest.ResponseRecorder {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		rec := doTranscription(engine, http.MethodGet, "/v1/audio/transcriptions/"+id, nil, "", "")
		status := ojson.GetBytes(rec.Body.Bytes(), "status").String()
		if rec.Code != http.StatusOK || (status != "queued" && status != "processing") {
			return rec
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %s still %s", id, status)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestAudioTranscriptionsAsyncCompleted(t *testing.T) {
	engine, captured := newTranscriptionTestServer(t, http.StatusOK, testDiarizedJSON)
	body, ct := transcriptionForm(t, map[string]string{"model": "SPARK-ASR", "response_format": "diarized_json"}, true)
	rec := doTranscription(engine, http.MethodPost, "/v1/audio/transcriptions", body, ct, "respond-async, wait=0")
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	id := ojson.GetBytes(rec.Body.Bytes(), "id").String()
	if id == "" || ojson.GetBytes(rec.Body.Bytes(), "status").String() != "queued" {
		t.Fatalf("bad 202 body: %s", rec.Body.String())
	}
	poll := waitTranscriptionJob(t, engine, id)
	out := poll.Body.Bytes()
	if poll.Code != http.StatusOK || ojson.GetBytes(out, "status").String() != "completed" || ojson.GetBytes(out, "id").String() != id {
		t.Fatalf("poll = %d %s", poll.Code, poll.Body.String())
	}
	if ojson.GetBytes(out, "segments.0.speaker").String() != "S01" || ojson.GetBytes(out, "text").String() != "hi there" {
		t.Fatalf("result not merged: %s", out)
	}
	// Results stay readable after being fetched.
	if again := doTranscription(engine, http.MethodGet, "/v1/audio/transcriptions/"+id, nil, "", ""); again.Code != http.StatusOK {
		t.Fatalf("second poll = %d", again.Code)
	}
	if n := len(captured()); n != 1 {
		t.Fatalf("upstream calls = %d", n)
	}
}

func TestAudioTranscriptionsAsyncFailed(t *testing.T) {
	engine, _ := newTranscriptionTestServer(t, http.StatusInternalServerError, `{"error":{"message":"model crashed"}}`)
	body, ct := transcriptionForm(t, map[string]string{"model": "SPARK-ASR"}, true)
	rec := doTranscription(engine, http.MethodPost, "/v1/audio/transcriptions", body, ct, "respond-async")
	id := ojson.GetBytes(rec.Body.Bytes(), "id").String()
	out := waitTranscriptionJob(t, engine, id).Body.Bytes()
	if ojson.GetBytes(out, "status").String() != "failed" || ojson.GetBytes(out, "error.message").String() == "" || ojson.GetBytes(out, "error.code").String() == "" {
		t.Fatalf("failed job body: %s", out)
	}
}

func TestAudioTranscriptionRetrieveUnknown(t *testing.T) {
	engine, _ := newTranscriptionTestServer(t, http.StatusOK, testDiarizedJSON)
	if rec := doTranscription(engine, http.MethodGet, "/v1/audio/transcriptions/transcr_nope", nil, "", ""); rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestTranscriptionJobStoreTTL(t *testing.T) {
	now := time.Date(2026, 11, 23, 0, 0, 0, 0, time.UTC)
	store := &transcriptionJobStore{jobs: map[string]*transcriptionJob{}, now: func() time.Time { return now }}
	done, _ := store.create("k1")
	pending, _ := store.create("k1")
	store.complete(done.id, []byte("plain text result"))
	failed, _ := store.create("k1")
	store.fail(failed.id, &interfaces.ErrorMessage{StatusCode: 413, Error: errors.New(`{"error":{"message":"audio longer than 300 seconds","code":"audio_too_long"}}`)})

	if body, ok := store.render(done.id, "k1"); !ok || ojson.GetBytes(body, "text").String() != "plain text result" {
		t.Fatalf("non-JSON result render = %s %v", body, ok)
	}
	if _, ok := store.render(done.id, "k2"); ok {
		t.Fatal("job visible to another API key")
	}
	if body, _ := store.render(failed.id, "k1"); ojson.GetBytes(body, "error.code").String() != "audio_too_long" ||
		ojson.GetBytes(body, "error.message").String() != "audio longer than 300 seconds" {
		t.Fatalf("failed render = %s", body)
	}
	now = now.Add(transcriptionJobTTL - time.Second)
	if _, ok := store.render(done.id, "k1"); !ok {
		t.Fatal("job evicted before TTL")
	}
	now = now.Add(2 * time.Second)
	if _, ok := store.render(done.id, "k1"); ok {
		t.Fatal("completed job kept after TTL")
	}
	if _, ok := store.render(failed.id, "k1"); ok {
		t.Fatal("failed job kept after TTL")
	}
	if _, ok := store.render(pending.id, "k1"); !ok {
		t.Fatal("unfinished job must not expire")
	}
}

func TestTranscriptionJobStoreLimitsActiveJobs(t *testing.T) {
	store := &transcriptionJobStore{jobs: map[string]*transcriptionJob{}, now: time.Now}
	var first *transcriptionJob
	for i := 0; i < maxActiveTranscriptionJobs; i++ {
		job, ok := store.create("k")
		if !ok {
			t.Fatalf("create %d rejected", i)
		}
		if first == nil {
			first = job
		}
	}
	if _, ok := store.create("k"); ok {
		t.Fatal("create beyond the active limit accepted")
	}
	store.complete(first.id, []byte(`{}`))
	if _, ok := store.create("k"); !ok {
		t.Fatal("finished job still counted as active")
	}
}

func TestPreferRespondAsync(t *testing.T) {
	for value, want := range map[string]bool{"respond-async": true, "wait=10, Respond-Async": true, "return=minimal": false, "": false} {
		header := http.Header{}
		if value != "" {
			header.Set("Prefer", value)
		}
		if got := preferRespondAsync(header); got != want {
			t.Fatalf("Prefer %q = %v, want %v", value, got, want)
		}
	}
}
